import os
import re
import time
import json
import random
import logging
import argparse
import csv
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, random_split

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
)
import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
#  1. CONFIGURATION
# =============================================================================

@dataclass
class FinetuneConfig:
    csv_path:        str   = "dataset/dataset.csv"
    audio_dir:       str   = "dataset/audio/"
    output_dir:      str   = "finetuned_model/"
    vocab_path:      str   = "config/vocab.json"
    base_model:      str   = "ai4bharat/indicwav2vec_v1_hindi"
    train_split:     float = 0.85
    seed:            int   = 42
    num_epochs:           int   = 30
    batch_size:           int   = 8
    eval_batch_size:      int   = 8
    gradient_accumulation:int   = 2
    learning_rate:        float = 1e-4
    warmup_steps:         int   = 500
    weight_decay:         float = 0.005
    max_grad_norm:        float = 1.0
    fp16:                 bool  = True
    save_steps:           int   = 200
    eval_steps:           int   = 200
    logging_steps:        int   = 50
    save_total_limit:     int   = 3
    sample_rate:          int   = 16000
    max_duration_sec:     float = 15.0
    min_duration_sec:     float = 0.5
    ctc_zero_infinity:    bool  = True
    attention_dropout:    float = 0.1
    hidden_dropout:       float = 0.1
    feat_proj_dropout:    float = 0.0
    mask_time_prob:       float = 0.05
    layerdrop:            float = 0.1
    freeze_feature_encoder: bool = True


# =============================================================================
#  2. TEXT CLEANING
# =============================================================================

HINDI_CHARS = re.compile(r"[^\u0900-\u097F\s]")

def clean_hindi_text(text: str) -> str:
    text = text.strip()
    text = HINDI_CHARS.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


# =============================================================================
#  3. VOCABULARY BUILDER
# =============================================================================

def build_vocab(csv_path: str, vocab_path: str):
    logger.info("Building vocabulary...")
    char_set = set()
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            char_set.update(list(clean_hindi_text(row["ground_truth"])))
    char_set.discard("")
    char_set.discard(" ")
    vocab = {c: i for i, c in enumerate(sorted(char_set))}
    vocab["[PAD]"] = len(vocab)
    vocab["[UNK]"] = len(vocab)
    vocab["|"]     = len(vocab)
    os.makedirs(os.path.dirname(vocab_path) or ".", exist_ok=True)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    logger.info(f"Vocabulary: {len(vocab)} characters → {vocab_path}")
    return vocab_path


# =============================================================================
#  4. DATASET
# =============================================================================

class HindiASRDataset(Dataset):
    def __init__(self, csv_path, audio_dir, processor, config):
        self.processor = processor
        self.audio_dir = audio_dir
        self.config    = config
        self.samples   = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                gt = clean_hindi_text(row["ground_truth"])
                if not gt:
                    continue
                ap = os.path.join(audio_dir, row["filename"].strip())
                if not os.path.exists(ap):
                    logger.warning(f"Missing: {ap}")
                    continue
                self.samples.append({"audio_path": ap, "ground_truth": gt})
        logger.info(f"Dataset: {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        waveform, sr = torchaudio.load(s["audio_path"])
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.config.sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.config.sample_rate)(waveform)
        waveform = waveform.squeeze().numpy()
        duration = len(waveform) / self.config.sample_rate
        if duration > self.config.max_duration_sec or duration < self.config.min_duration_sec:
            return None
        return {"audio": waveform, "text": s["ground_truth"].replace(" ", "|")}


# =============================================================================
#  5. CTC DATA COLLATOR
# =============================================================================

@dataclass
class CTCDataCollator:
    processor: Wav2Vec2Processor
    padding:   Union[bool, str] = True

    def __call__(self, features):
        features = [f for f in features if f is not None]
        if not features:
            return None
        input_values = self.processor(
            [f["audio"] for f in features],
            sampling_rate=16000, return_tensors="pt", padding=self.padding,
        ).input_values
        with self.processor.as_target_processor():
            labels_batch = self.processor(
                [f["text"] for f in features],
                return_tensors="pt", padding=self.padding,
            )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)
        return {"input_values": input_values, "labels": labels}


# =============================================================================
#  6. WER METRIC
# =============================================================================

wer_metric = evaluate.load("wer")

def compute_metrics(pred, processor):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str  = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    logger.info(f"WER: {wer:.4f}")
    return {"wer": wer}


# =============================================================================
#  7. PROCESSOR + MODEL
# =============================================================================

def build_processor(config: FinetuneConfig) -> Wav2Vec2Processor:
    if not os.path.exists(config.vocab_path):
        build_vocab(config.csv_path, config.vocab_path)
    tokenizer = Wav2Vec2CTCTokenizer(
        config.vocab_path, unk_token="[UNK]", pad_token="[PAD]",
        word_delimiter_token="|")
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=config.sample_rate,
        padding_value=0.0, do_normalize=True, return_attention_mask=True)
    return Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def build_model(config: FinetuneConfig, processor: Wav2Vec2Processor) -> Wav2Vec2ForCTC:
    logger.info(f"Loading: {config.base_model}")
    model = Wav2Vec2ForCTC.from_pretrained(
        config.base_model,
        attention_dropout=config.attention_dropout,
        hidden_dropout=config.hidden_dropout,
        feat_proj_dropout=config.feat_proj_dropout,
        mask_time_prob=config.mask_time_prob,
        layerdrop=config.layerdrop,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ctc_zero_infinity=config.ctc_zero_infinity,
        ignore_mismatched_sizes=True,
    )
    if config.freeze_feature_encoder:
        model.freeze_feature_encoder()
        logger.info("CNN encoder frozen.")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {trainable:,}")
    return model


# =============================================================================
#  8. TRAINING
# =============================================================================

def finetune(config: FinetuneConfig):
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    processor = build_processor(config)
    full_ds   = HindiASRDataset(config.csv_path, config.audio_dir, processor, config)
    n_train   = int(len(full_ds) * config.train_split)
    n_eval    = len(full_ds) - n_train
    train_ds, eval_ds = random_split(
        full_ds, [n_train, n_eval],
        generator=torch.Generator().manual_seed(config.seed))
    logger.info(f"Train: {n_train} | Eval: {n_eval}")

    collator = CTCDataCollator(processor=processor)
    model    = build_model(config, processor)

    training_args = TrainingArguments(
        output_dir                  = config.output_dir,
        group_by_length             = True,
        per_device_train_batch_size = config.batch_size,
        per_device_eval_batch_size  = config.eval_batch_size,
        gradient_accumulation_steps = config.gradient_accumulation,
        evaluation_strategy         = "steps",
        num_train_epochs            = config.num_epochs,
        fp16                        = config.fp16 and torch.cuda.is_available(),
        save_steps                  = config.save_steps,
        eval_steps                  = config.eval_steps,
        logging_steps               = config.logging_steps,
        learning_rate               = config.learning_rate,
        warmup_steps                = config.warmup_steps,
        weight_decay                = config.weight_decay,
        max_grad_norm               = config.max_grad_norm,
        save_total_limit            = config.save_total_limit,
        load_best_model_at_end      = True,
        metric_for_best_model       = "wer",
        greater_is_better           = False,
        report_to                   = ["none"],
        dataloader_num_workers      = 4,
        seed                        = config.seed,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        tokenizer=processor.feature_extractor,
        data_collator=collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor),
    )

    logger.info("=" * 60)
    logger.info("  VaakSetu — Finetuning on Yotta / DGX4 GPU Server")
    logger.info(f"  Epochs     : {config.num_epochs}")
    logger.info(f"  Batch size : {config.batch_size} × {config.gradient_accumulation}")
    logger.info(f"  LR         : {config.learning_rate}")
    logger.info(f"  FP16       : {config.fp16 and torch.cuda.is_available()}")
    logger.info("=" * 60)

    t0 = time.time()
    trainer.train()
    elapsed = round(time.time() - t0, 2)
    logger.info(f"Done — {elapsed:.1f}s ({elapsed/60:.1f} min)")

    final_path = os.path.join(config.output_dir, "final")
    trainer.save_model(final_path)
    processor.save_pretrained(final_path)
    logger.info(f"Model saved → {final_path}")

    results = trainer.evaluate()
    logger.info(f"Final WER: {results.get('eval_wer', 'N/A'):.4f}")
    with open(os.path.join(config.output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# =============================================================================
#  CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="VaakSetu Finetuning")
    p.add_argument("--csv",        type=str, default="dataset/dataset.csv")
    p.add_argument("--audio-dir",  type=str, default="dataset/audio/")
    p.add_argument("--output-dir", type=str, default="finetuned_model/")
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--no-fp16",    action="store_true")
    p.add_argument("--cpu",        action="store_true")
    p.add_argument("--vocab-only", action="store_true")
    args = p.parse_args()

    config = FinetuneConfig()
    config.csv_path      = args.csv
    config.audio_dir     = args.audio_dir
    config.output_dir    = args.output_dir
    config.num_epochs    = args.epochs
    config.batch_size    = args.batch_size
    config.learning_rate = args.lr
    config.fp16          = not args.no_fp16
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.vocab_only:
        build_vocab(config.csv_path, config.vocab_path)
        return
    finetune(config)


if __name__ == "__main__":
    main()
