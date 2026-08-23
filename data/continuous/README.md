# data/continuous/README.md

## What is here

This directory contains **synthetic landmark sequences** generated from the W7CH
single-sign image dataset. They are used to validate the V2 temporal architecture
(BiLSTM/GRU + Attention) before real LST video data is available.

## ⚠ What this is NOT

These sequences are **NOT** real LST phrases. They are assembled by sampling
single-sign features and concatenating them into variable-length temporal
sequences with noise augmentation. No video, no inter-sign transition, no
linguistic structure.

**Performance on this data must never be presented as LST phrase recognition
accuracy.**

## Directory structure

```
data/continuous/
├── landmarks/          # .npy files, each (T, 126) float32
├── videos/             # (placeholder for future real videos)
├── metadata.csv        # per-sample metadata
└── README.md           # this file
```

## metadata.csv schema

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | str | Unique identifier, e.g. `seq_00142` |
| `speaker_id` | str | Pseudo-speaker, e.g. `spk_0` to `spk_4` |
| `label` | str | Sign name (e.g. `bou`, `om`) |
| `label_idx` | int | Integer class index |
| `length` | int | Number of frames in the sequence |
| `split` | str | `train`, `val`, or `test` |
| `source` | str | Always `synthetic_w7ch` for this dataset |

## Speaker-independent split

| Split | Speakers | Approx % |
|-------|----------|----------|
| Train | spk_0, spk_1, spk_2 | 60% |
| Val   | spk_3                | 20% |
| Test  | spk_4                | 20% |

## Vocabulary (8 signs)

`bou`, `om`, `labes`, `non`, `oui`, `dar`, `thnin`, `mar7ba`

Selected from W7CH for being sufficiently distinct and potentially useful in
common expressions.

## Regenerating

```bash
python scripts/gen_synthetic_landmarks.py
```

Requires `data/processed/features.npy` and `data/processed/labels.npy`
(prepared by `scripts/prepare_data.py` from V1).
