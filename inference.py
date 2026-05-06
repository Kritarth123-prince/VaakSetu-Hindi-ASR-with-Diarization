import os
import re
import time
import json
import argparse
import logging
from datetime import datetime, timedelta

import torch
import torchaudio
from tqdm import tqdm
from omegaconf import OmegaConf
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from fairseq.data.data_utils import post_process
from fairseq.tasks.audio_finetuning import AudioFinetuningTask
from fairseq import checkpoint_utils
from fairseq.data.dictionary import Dictionary

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from pyannote.audio import Pipeline as DiarizationPipeline
    DIARIZATION_AVAILABLE = True
except ImportError:
    DIARIZATION_AVAILABLE = False
    logging.warning("pyannote.audio not installed — speaker diarization disabled.")

try:
    from transformers import pipeline as hf_pipeline
    PUNCTUATION_AVAILABLE = True
except ImportError:
    PUNCTUATION_AVAILABLE = False
    logging.warning("transformers not installed — punctuation restoration disabled.")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Global caches ─────────────────────────────────────────────────────────────
transcribed_files  = set()
model_cache        = None   # (model, task, grapheme_dict)
diarization_cache  = None   # pyannote pipeline
punctuation_cache  = None   # HuggingFace punctuation pipeline

SUPPORTED_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


# =============================================================================
#  1. FILE READINESS
# =============================================================================

def is_file_fully_copied(file_path, check_interval=1, max_checks=6):
    """
    Poll file size until stable — ensures the file is fully written
    before we start processing it.
    """
    previous_size = -1
    for _ in range(max_checks):
        current_size = os.path.getsize(file_path)
        if current_size == previous_size:
            return True
        previous_size = current_size
        time.sleep(check_interval)
        check_interval = min(check_interval * 2, 8)
    return False


# =============================================================================
#  2. AUDIO UTILITIES
# =============================================================================

def resample_audio(audio_path, target_sr=16000):
    """Resample audio in-place to target_sr (default 16 kHz)."""
    waveform, orig_sr = torchaudio.load(audio_path)
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        waveform  = resampler(waveform)
        torchaudio.save(audio_path, waveform, target_sr)
        logger.info(f"Resampled {os.path.basename(audio_path)}: {orig_sr} Hz → {target_sr} Hz")
    return audio_path


def split_audio(audio_path, max_duration=30, chunks_dir="chunks"):
    """
    Split audio into ≤max_duration second chunks.
    Returns list of (chunk_path, waveform, start_sec, end_sec).
    """
    os.makedirs(chunks_dir, exist_ok=True)
    waveform, sr     = torchaudio.load(audio_path)
    total_duration   = waveform.size(1) / sr

    if total_duration <= max_duration:
        return [(audio_path, waveform, 0.0, total_duration)]

    chunks     = []
    chunk_size = int(max_duration * sr)
    base_name  = os.path.splitext(os.path.basename(audio_path))[0]

    for idx, start in enumerate(range(0, waveform.size(1), chunk_size)):
        end            = min(start + chunk_size, waveform.size(1))
        chunk_waveform = waveform[:, start:end]
        chunk_path     = os.path.join(chunks_dir, f"{base_name}_chunk_{idx}.wav")
        torchaudio.save(chunk_path, chunk_waveform, sr)
        chunks.append((chunk_path, chunk_waveform, start / sr, end / sr))

    logger.info(f"Split into {len(chunks)} chunks ({total_duration:.1f}s total).")
    return chunks


def preprocess_audio(waveform, sample_rate, task):
    """Zero-mean, unit-max normalisation."""
    waveform = waveform - waveform.mean()
    max_val  = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val
    return waveform


# =============================================================================
#  3. MODEL LOADING (cached)
# =============================================================================

def load_model_and_dict(config_path, checkpoint_path, dictionary_path, use_cuda=True):
    """Load finetuned wav2vec 2.0 model + grapheme dictionary. Cached after first call."""
    global model_cache
    if model_cache is None:
        logger.info("Loading ASR model (first call — this may take a moment)...")
        config  = OmegaConf.load(config_path)
        task    = AudioFinetuningTask.setup_task(config.task)
        models, _, _ = checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
        model   = models[0]
        model.eval()

        if use_cuda and torch.cuda.is_available():
            model = model.cuda()
            logger.info("Model loaded on GPU.")
        else:
            logger.info("Model loaded on CPU.")

        grapheme_dict = Dictionary.load(dictionary_path)
        model_cache   = (model, task, grapheme_dict)

    return model_cache


# =============================================================================
#  4. SPEAKER DIARIZATION
# =============================================================================

def load_diarization_pipeline(hf_token=None):
    """Load pyannote.audio 3.1 speaker diarization pipeline (cached)."""
    global diarization_cache
    if diarization_cache is None and DIARIZATION_AVAILABLE:
        logger.info("Loading speaker diarization pipeline...")
        try:
            diarization_cache = DiarizationPipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
            if torch.cuda.is_available():
                diarization_cache = diarization_cache.to(torch.device("cuda"))
            logger.info("Diarization pipeline ready.")
        except Exception as e:
            logger.warning(f"Diarization pipeline failed to load: {e}")
    return diarization_cache


def diarize_audio(audio_path, hf_token=None):
    """
    Run speaker diarization on an audio file.
    Returns list of {start, end, speaker} dicts.
    """
    pipeline = load_diarization_pipeline(hf_token)
    if pipeline is None:
        return []
    try:
        result   = pipeline(audio_path)
        segments = []
        for turn, _, speaker in result.itertracks(yield_label=True):
            segments.append({
                "start":   round(turn.start, 3),
                "end":     round(turn.end,   3),
                "speaker": speaker,
            })
        n_speakers = len(set(s["speaker"] for s in segments))
        logger.info(f"Diarization: {n_speakers} speaker(s) detected.")
        return segments
    except Exception as e:
        logger.warning(f"Diarization failed: {e}")
        return []


def assign_speaker(diarization_segments, chunk_start, chunk_end):
    """Return the speaker with maximum overlap in the given time range."""
    if not diarization_segments:
        return "SPEAKER_00"
    speaker_time = {}
    for seg in diarization_segments:
        overlap = max(0.0, min(seg["end"], chunk_end) - max(seg["start"], chunk_start))
        if overlap > 0:
            speaker_time[seg["speaker"]] = speaker_time.get(seg["speaker"], 0) + overlap
    return max(speaker_time, key=speaker_time.get) if speaker_time else "SPEAKER_00"


# =============================================================================
#  5. CTC DECODING + CONFIDENCE SCORING
# =============================================================================

def ctc_decode_with_confidence(logits, dictionary):
    """
    Greedy CTC decode with per-segment confidence score.
    logits : tensor of shape (T, vocab) or (1, T, vocab) — log-probabilities
    Returns (decoded_tokens, confidence_float)
    """
    if logits.dim() == 3:
        logits = logits.squeeze(0)

    probs                   = torch.exp(logits)              # log-probs → probs
    best_probs, best_tokens = probs.max(dim=-1)              # (T,)
    confidence              = best_probs.mean().item()       # avg per-frame confidence

    tokens_list = best_tokens.squeeze().tolist()
    if isinstance(tokens_list, int):
        tokens_list = [tokens_list]

    decoded, prev = [], None
    for token in tokens_list:
        if token != prev and token != dictionary.pad():
            decoded.append(token)
        prev = token

    return decoded, round(confidence, 4)


def tokens_to_string(tokens, dictionary):
    """Convert token index list → readable string."""
    return post_process(dictionary.string(tokens), symbol='letter')


# =============================================================================
#  6. PUNCTUATION RESTORATION
# =============================================================================

def load_punctuation_model():
    """Load multilingual punctuation restoration model (cached)."""
    global punctuation_cache
    if punctuation_cache is None and PUNCTUATION_AVAILABLE:
        logger.info("Loading punctuation restoration model...")
        try:
            punctuation_cache = hf_pipeline(
                "token-classification",
                model="oliverguhr/fullstop-punctuation-multilang-large",
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1,
            )
            logger.info("Punctuation model ready.")
        except Exception as e:
            logger.warning(f"Punctuation model failed to load: {e}")
    return punctuation_cache


def restore_punctuation(text):
    """Insert commas, periods, question marks and capitalise sentences."""
    model = load_punctuation_model()
    if model is None or not text.strip():
        return text
    try:
        results = model(text)
        output  = ""
        for item in results:
            word  = item["word"].replace("▁", " ").strip()
            label = item["entity_group"]
            if   label == "PERIOD":   output += word + ". "
            elif label == "COMMA":    output += word + ", "
            elif label == "QUESTION": output += word + "? "
            else:                     output += word + " "
        sentences = re.split(r'(?<=[.!?])\s+', output.strip())
        return " ".join(s.capitalize() for s in sentences if s)
    except Exception as e:
        logger.warning(f"Punctuation restoration failed: {e}")
        return text


# =============================================================================
#  7. SUBTITLE WRITERS  (SRT + VTT)
# =============================================================================

def _to_srt_ts(sec):
    td   = timedelta(seconds=sec)
    tot  = int(td.total_seconds())
    ms   = int((td.total_seconds() - tot) * 1000)
    h, m, s = tot // 3600, (tot % 3600) // 60, tot % 60
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def _to_vtt_ts(sec):
    return _to_srt_ts(sec).replace(",", ".")


def write_srt(segments, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{_to_srt_ts(seg['start'])} --> {_to_srt_ts(seg['end'])}\n")
            f.write(f"[{seg['speaker']}] {seg['text']}\n\n")
    logger.info(f"SRT → {path}")


def write_vtt(segments, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{_to_vtt_ts(seg['start'])} --> {_to_vtt_ts(seg['end'])}\n")
            f.write(f"[{seg['speaker']}] {seg['text']}\n\n")
    logger.info(f"VTT → {path}")


# =============================================================================
#  8. CORE TRANSCRIPTION
# =============================================================================

def transcribe_audio(
    config_path,
    checkpoint_path,
    dictionary_path,
    audio_path,
    use_cuda=True,
    hf_token=None,
    enable_diarization=True,
    enable_punctuation=True,
):
    """
    Full pipeline for a single audio file.
    Outputs: .txt  .srt  .vtt  .json  inside transcripts/
    Returns path to .txt output or None on failure.
    """
    t0        = time.time()
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    out_dir   = "transcripts"
    os.makedirs(out_dir, exist_ok=True)

    try:
        # ── Stage 1: File readiness ───────────────────────────────────────
        if not is_file_fully_copied(audio_path):
            logger.warning(f"File still being written — skipping: {audio_path}")
            return None

        # ── Stage 2: Resample ─────────────────────────────────────────────
        audio_path = resample_audio(audio_path)

        # ── Stage 3: Speaker Diarization ──────────────────────────────────
        diar_segs = []
        if enable_diarization:
            diar_segs = diarize_audio(audio_path, hf_token=hf_token)

        # ── Stage 4: Load model ───────────────────────────────────────────
        model, task, grapheme_dict = load_model_and_dict(
            config_path, checkpoint_path, dictionary_path, use_cuda
        )

        # ── Stage 5: Chunk audio ──────────────────────────────────────────
        chunks = split_audio(audio_path)

        # ── Stage 6: CTC inference + confidence scoring ───────────────────
        results = []
        for chunk_path, waveform, c_start, c_end in chunks:
            waveform = preprocess_audio(
                waveform, torchaudio.info(chunk_path).sample_rate, task
            )
            if use_cuda and torch.cuda.is_available():
                waveform = waveform.cuda()

            net_input = {
                "source":       waveform,
                "padding_mask": None,
                "src_lengths":  torch.tensor([waveform.size(1)]),
            }
            if use_cuda and torch.cuda.is_available():
                net_input["src_lengths"] = net_input["src_lengths"].cuda()

            with torch.no_grad():
                net_output      = model(**net_input)
                grapheme_lprobs = model.get_normalized_probs(net_output, log_probs=True)

            decoded, confidence = ctc_decode_with_confidence(grapheme_lprobs, grapheme_dict)
            raw_text            = tokens_to_string(decoded, grapheme_dict)
            speaker             = assign_speaker(diar_segs, c_start, c_end)

            results.append({
                "start":      c_start,
                "end":        c_end,
                "speaker":    speaker,
                "text":       raw_text,
                "confidence": confidence,
            })

            if chunk_path != audio_path:
                os.remove(chunk_path)

        # ── Stage 7: Punctuation restoration ─────────────────────────────
        if enable_punctuation:
            for seg in results:
                seg["text"] = restore_punctuation(seg["text"])

        # ── Stage 8: Write outputs ────────────────────────────────────────
        overall_conf = (
            sum(s["confidence"] for s in results) / len(results) if results else 0.0
        )
        n_speakers = len(set(s["speaker"] for s in results))

        # .txt
        txt_path = os.path.join(out_dir, f"{base_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"File       : {audio_path}\n")
            f.write(f"Speakers   : {n_speakers}\n")
            f.write(f"Confidence : {overall_conf:.4f}\n")
            f.write("=" * 60 + "\n")
            for seg in results:
                f.write(
                    f"[{seg['speaker']}] "
                    f"({seg['start']:.2f}s–{seg['end']:.2f}s) "
                    f"[conf:{seg['confidence']:.2f}] "
                    f"{seg['text']}\n"
                )

        # .srt / .vtt
        srt_path = os.path.join(out_dir, f"{base_name}.srt")
        vtt_path = os.path.join(out_dir, f"{base_name}.vtt")
        write_srt(results, srt_path)
        write_vtt(results, vtt_path)

        # .json
        elapsed  = round(time.time() - t0, 2)
        metadata = {
            "file":             audio_path,
            "timestamp":        datetime.now().isoformat(),
            "overall_confidence": overall_conf,
            "elapsed_seconds":  elapsed,
            "num_speakers":     n_speakers,
            "num_chunks":       len(results),
            "segments":         results,
        }
        json_path = os.path.join(out_dir, f"{base_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # ── Console summary ───────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info(f"FILE       : {audio_path}")
        logger.info(f"SPEAKERS   : {n_speakers}")
        logger.info(f"CONFIDENCE : {overall_conf:.4f}")
        logger.info(f"TIME       : {elapsed}s")
        logger.info(f"OUTPUTS    : .txt  .srt  .vtt  .json  → {out_dir}/")
        logger.info("=" * 60)
        print("\n--- TRANSCRIPT ---")
        for seg in results:
            print(f"[{seg['speaker']}] {seg['text']}")
        print("------------------\n")

        return txt_path

    except RuntimeError as e:
        if "Invalid data found when processing input" in str(e):
            logger.error(f"Corrupt/invalid audio, retrying in 5s: {audio_path}")
            time.sleep(5)
            return transcribe_audio(
                config_path, checkpoint_path, dictionary_path,
                audio_path, use_cuda, hf_token,
                enable_diarization, enable_punctuation,
            )
        logger.error(f"Unhandled RuntimeError: {e}")
        raise


# =============================================================================
#  9. BATCH PROCESSING
# =============================================================================

def batch_transcribe(
    config_path, checkpoint_path, dictionary_path,
    input_dir, use_cuda=True, hf_token=None,
    enable_diarization=True, enable_punctuation=True,
):
    """Process all supported audio files in input_dir with a progress bar."""
    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    ]
    if not files:
        logger.warning(f"No supported audio files found in {input_dir}")
        return

    logger.info(f"Batch mode: {len(files)} file(s) found.")
    load_model_and_dict(config_path, checkpoint_path, dictionary_path, use_cuda)

    success = failed = 0
    for audio_path in tqdm(files, desc="Transcribing", unit="file"):
        result = transcribe_audio(
            config_path, checkpoint_path, dictionary_path,
            audio_path, use_cuda, hf_token,
            enable_diarization, enable_punctuation,
        )
        if result:
            success += 1
        else:
            failed += 1

    logger.info(f"Batch complete — ✓ {success} succeeded | ✗ {failed} failed.")


# =============================================================================
#  10. REAL-TIME FOLDER MONITORING (watchdog)
# =============================================================================

class AudioHandler(FileSystemEventHandler):
    """Watches a directory and auto-transcribes any new audio file."""

    def __init__(
        self, config_path, checkpoint_path, dictionary_path,
        use_cuda=True, hf_token=None,
        enable_diarization=True, enable_punctuation=True,
    ):
        self.config_path        = config_path
        self.checkpoint_path    = checkpoint_path
        self.dictionary_path    = dictionary_path
        self.use_cuda           = use_cuda
        self.hf_token           = hf_token
        self.enable_diarization = enable_diarization
        self.enable_punctuation = enable_punctuation

    def on_created(self, event):
        if event.is_directory:
            return
        audio_path = event.src_path
        if os.path.splitext(audio_path)[1].lower() not in SUPPORTED_EXT:
            return
        if audio_path in transcribed_files:
            logger.info(f"Already transcribed: {audio_path}")
            return
        logger.info(f"New file detected: {audio_path}")
        transcribe_audio(
            self.config_path, self.checkpoint_path, self.dictionary_path,
            audio_path, self.use_cuda, self.hf_token,
            self.enable_diarization, self.enable_punctuation,
        )
        transcribed_files.add(audio_path)


# =============================================================================
#  CLI + MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="VaakSetu — Hindi ASR Inference Pipeline"
    )
    p.add_argument("--batch",            action="store_true",
                   help="Process all files in input-dir at once.")
    p.add_argument("--no-diarization",   action="store_true",
                   help="Disable speaker diarization.")
    p.add_argument("--no-punctuation",   action="store_true",
                   help="Disable punctuation restoration.")
    p.add_argument("--cpu",              action="store_true",
                   help="Force CPU inference.")
    p.add_argument("--hf-token",         type=str, default=None,
                   help="HuggingFace token for pyannote diarization model.")
    p.add_argument("--input-dir",        type=str,
                   default="/raid/ganesh/pdadiga/suryansh/w2v2-txt-transcription/input/")
    p.add_argument("--config",           type=str,
                   default="/raid/ganesh/pdadiga/suryansh/w2v2-txt-transcription/config/ai4b_xlsr.yaml")
    p.add_argument("--dictionary",       type=str,
                   default="/raid/ganesh/pdadiga/suryansh/w2v2-txt-transcription/config/dic.ltr.txt")
    p.add_argument("--checkpoint",       type=str,
                   default="/raid/ganesh/pdadiga/suryansh/w2v2-txt-transcription/model/checkpoint_best.pt")
    return p.parse_args()


def main():
    args = parse_args()

    config_path     = args.config
    dictionary_path = args.dictionary
    checkpoint_path = args.checkpoint
    input_dir       = args.input_dir
    use_cuda        = not args.cpu
    hf_token        = args.hf_token
    enable_diar     = not args.no_diarization
    enable_punc     = not args.no_punctuation

    logger.info("=" * 60)
    logger.info("  VaakSetu — Hindi ASR Inference Pipeline")
    logger.info(f"  Mode        : {'BATCH' if args.batch else 'MONITOR'}")
    logger.info(f"  GPU         : {use_cuda and torch.cuda.is_available()}")
    logger.info(f"  Diarization : {enable_diar}")
    logger.info(f"  Punctuation : {enable_punc}")
    logger.info("=" * 60)

    if args.batch:
        batch_transcribe(
            config_path, checkpoint_path, dictionary_path,
            input_dir, use_cuda, hf_token, enable_diar, enable_punc,
        )
    else:
        load_model_and_dict(config_path, checkpoint_path, dictionary_path, use_cuda)
        handler  = AudioHandler(
            config_path, checkpoint_path, dictionary_path,
            use_cuda, hf_token, enable_diar, enable_punc,
        )
        observer = Observer()
        observer.schedule(handler, input_dir, recursive=False)
        try:
            observer.start()
            logger.info(f"Monitoring {input_dir} — press Ctrl+C to stop.")
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            logger.info("Stopped.")
        observer.join()


if __name__ == "__main__":
    main()
