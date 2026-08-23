#!/usr/bin/env python3
"""Generate synthetic landmark sequences from W7CH features for V2 pipeline testing.

IMPORTANT: These sequences are NOT real LST phrases. They are assembled from
isolated single-sign landmarks sampled from the W7CH image dataset and
concatenated/repeated to form variable-length temporal sequences. Their purpose
is strictly technical validation of the V2 temporal architecture (padding,
masking, attention, training loop). Performance on these sequences must always
be reported separately from real LST results.

Usage:
    python scripts/gen_synthetic_landmarks.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.settings import (
    PROCESSED_DATA_DIR,
    CONTINUOUS_DATA_DIR,
    CONTINUOUS_LANDMARKS_DIR,
    SYNTHETIC_BENCHMARK_VOCAB,
    SYNTHETIC_BENCHMARK_NUM_SPEAKERS,
    SYNTHETIC_BENCHMARK_SEQ_LENGTHS,
    SYNTHETIC_BENCHMARK_SAMPLES_PER_SLICE,
)


def load_w7ch_pool():
    features = np.load(PROCESSED_DATA_DIR / "features.npy")
    labels = np.load(PROCESSED_DATA_DIR / "labels.npy")
    with open(PROCESSED_DATA_DIR / "class_names.txt", encoding="utf-8") as f:
        class_names = [line.strip() for line in f if line.strip()]
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    return features, labels, class_names, class_to_idx


def assign_speakers(features, labels, class_to_idx, vocab, num_speakers):
    vocab_indices = [class_to_idx[name] for name in vocab]
    pool = {}
    for cls_idx in vocab_indices:
        mask = labels == cls_idx
        indices = np.where(mask)[0]
        per_speaker = [[] for _ in range(num_speakers)]
        for i, idx in enumerate(indices):
            per_speaker[i % num_speakers].append(int(idx))
        pool[cls_idx] = per_speaker
    return pool


def sample_sequence(
    pool: dict,
    cls_idx: int,
    speaker_id: int,
    seq_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    available = pool[cls_idx][speaker_id]
    chosen_indices = rng.choice(available, size=seq_len, replace=True)
    frames = np.stack([np.load(f"{CONTINUOUS_LANDMARKS_DIR.parent}/features_cache.npy")[i]
                       if False else _load_features()[i]
                       for i in chosen_indices], axis=0)
    return frames


_features_cache = None

def _load_features():
    global _features_cache
    if _features_cache is None:
        _features_cache = np.load(PROCESSED_DATA_DIR / "features.npy")
    return _features_cache


def generate_synthetic_dataset():
    features, labels, class_names, class_to_idx = load_w7ch_pool()
    features = features.astype(np.float32)
    vocab = SYNTHETIC_BENCHMARK_VOCAB
    num_speakers = SYNTHETIC_BENCHMARK_NUM_SPEAKERS
    seq_lengths = SYNTHETIC_BENCHMARK_SEQ_LENGTHS
    samples_per_slice = SYNTHETIC_BENCHMARK_SAMPLES_PER_SLICE
    rng = np.random.default_rng(42)

    vocab_contiguous = {name: i for i, name in enumerate(vocab)}
    pool = {}
    for name in vocab:
        cls_idx = class_to_idx[name]
        mask = labels == cls_idx
        indices = np.where(mask)[0]
        per_speaker = [[] for _ in range(num_speakers)]
        for i, idx in enumerate(indices):
            per_speaker[i % num_speakers].append(int(idx))
        pool[cls_idx] = per_speaker

    # assign splits: speakers 0-2 train, 3 val, 4 test
    split_map = {0: "train", 1: "train", 2: "train", 3: "val", 4: "test"}
    landmarks_dir = CONTINUOUS_DATA_DIR / "landmarks"
    landmarks_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    seq_id = 0

    for name in vocab:
        cls_idx = class_to_idx[name]
        contiguous_idx = vocab_contiguous[name]
        for spk in range(num_speakers):
            split = split_map[spk]
            for seq_len in seq_lengths:
                for s in range(samples_per_slice):
                    chosen = rng.choice(pool[cls_idx][spk], size=seq_len, replace=True)
                    seq_frames = features[chosen].copy()
                    noise = rng.normal(0, 0.02, seq_frames.shape).astype(np.float32)
                    seq_frames = np.clip(seq_frames + noise, -5.0, 5.0).astype(np.float32)
                    fname = f"seq_{seq_id:05d}.npy"
                    np.save(landmarks_dir / fname, seq_frames)
                    metadata_rows.append({
                        "sample_id": f"seq_{seq_id:05d}",
                        "speaker_id": f"spk_{spk}",
                        "label": name,
                        "label_idx": contiguous_idx,
                        "length": seq_len,
                        "split": split,
                        "source": "synthetic_w7ch",
                    })
                    seq_id += 1

    meta_path = CONTINUOUS_DATA_DIR / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "speaker_id", "label", "label_idx", "length", "split", "source"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"Generated {seq_id} synthetic sequences")
    print(f"  Vocab: {len(vocab)} signs")
    print(f"  Speakers: {num_speakers} pseudo-speakers")
    print(f"  Seq lengths: {seq_lengths}")
    print(f"  Split: train={sum(1 for r in metadata_rows if r['split']=='train')}, "
          f"val={sum(1 for r in metadata_rows if r['split']=='val')}, "
          f"test={sum(1 for r in metadata_rows if r['split']=='test')}")
    print(f"  Saved to: {CONTINUOUS_DATA_DIR}")
    print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    generate_synthetic_dataset()
