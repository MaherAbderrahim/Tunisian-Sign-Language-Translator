# V2 Report — Continuous LST Recognition Prototype

> **Date:** 2026-08-19 | **Status:** Prototype (synthetic only) | **Real LST:** Pending data acquisition

## Purpose

Document the V2 continuous sign recognition infrastructure built on top of the existing V1 isolated-sign classifier. All V1 code remains **untouched**. This document clarifies what is real, what is synthetic, and what remains to be done.

---

## What was implemented

### 1. Preprocessing (`src/preprocessing/`)

| File | Purpose |
|------|---------|
| `normalization.py` | Wrist-centric per-frame normalization (V1-compatible range) |
| `landmarks.py` | Video → (T, 126) landmark extraction using the existing V1 `MediaPipeLandmarkExtractor` |
| `temporal.py` | Velocity, acceleration, fixed-length padding, `collate_padded` for DataLoader |

**V1 compatibility:** V1's `normalize_features()` is not modified. The new `normalize_frame_wrist()` follows the same [-5, 5] range convention so V1 classifiers work on V2-normalized input.

### 2. Temporal Models (`src/models/continuous/`)

| File | Purpose |
|------|---------|
| `base.py` | Abstract `ContinuousSignModel` with `encode` / `forward` / `predict` / `predict_proba` / `predict_topk` |
| `attention.py` | Additive temporal attention pooling: `(B, T, H) → (B, H)` |
| `bilstm.py` | **BiLSTM + Attention** — primary baseline |
| `gru.py` | **GRU + Attention** — comparison baseline |
| `v1_adapter.py` | Wraps the V1 `TSLClassifier` for per-frame inference in sequences (V1 itself is frozen) |

**V1 reuse:** `V1SequenceAdapter` loads `best_model.pth` and runs inference per-frame without modifying V1 weights.

### 3. Pipeline (`src/continuous_translation/`)

| File | Purpose |
|------|---------|
| `pipeline.py` | `ContinuousSignPipeline`: video file → landmarks → normalization → V1 per-frame + temporal model prediction |
| `inference.py` | `ContinuousInferenceEngine`: real-time buffer-based inference with majority-vote smoothing |

### 4. Synthetic Benchmark (`data/continuous/`)

| Path | Description |
|------|-------------|
| `data/continuous/landmarks/` | `.npy` files, each `(T, 126)` |
| `data/continuous/metadata.csv` | `sample_id, speaker_id, label, label_idx, length, split, source` |
| `data/continuous/README.md` | Explains what is and is not |

---

## ⚠ What is synthetic (NOT real LST)

The synthetic benchmark was generated from **isolated single-sign images** (W7CH dataset) by:

1. Selecting 8 vocabulary signs from W7CH
2. Grouping 10,600 features (200 per class) across 5 pseudo-speakers (modulo on index)
3. For each (sign, speaker, length) slice, sampling N frames from that speaker's pool
4. Repeating/sampling with jitter to form variable-length sequences

**This is NOT a video.** There is no temporal transition between signs. It is a concatenation of single-sign features with noise, designed to verify the pipeline works end-to-end.

**Performance on this benchmark is a technical metric, not an LST recognition metric.**

---

## Benchmark structure

| Parameter | Value |
|-----------|-------|
| Vocabulary | 8 signs (bou, om, labes, non, oui, dar, thnin, mar7ba) |
| Pseudo-speakers | 5 |
| Sequence lengths | 3, 5, 7, 10 frames |
| Samples per slice | 15 |
| **Total sequences** | 8 × 5 × 4 × 15 = **2,400** |
| Train speakers | 0, 1, 2 (60%) |
| Val speaker | 3 (20%) |
| Test speaker | 4 (20%) |
| **Split** | Speaker-independent |

---

## Results

### BiLSTM + Attention (740k params)

| Metric | Value |
|--------|-------|
| Accuracy | **0.9979** |
| Macro F1 | **0.9979** |
| Val accuracy (best) | 0.998 |
| Epochs to converge | 13 (early stop) |
| Train time (CPU) | ~15s |

### GRU + Attention (223k params)

| Metric | Value |
|--------|-------|
| Accuracy | **0.9979** |
| Macro F1 | **0.9979** |
| Val accuracy (best) | 0.998 |
| Epochs to converge | 16 (early stop) |
| Train time (CPU) | ~15s |

### V1 → Sequence (baseline)

| Metric | Value |
|--------|-------|
| Method | Per-frame V1 + majority vote (no temporal learning) |
| Accuracy | (pending real data test) |

> **Note:** Near-perfect scores on the synthetic benchmark are expected because the task is
> trivial (classify which of 8 signs is repeated in the sequence). This confirms the
> architecture works; it does NOT indicate real LST recognition capability.

---

## Two proposed pipelines

### Pipeline 1: V1 → Sign Sequence
```
Video → MediaPipe → Landmarks (T,126) → V1 classifier (per-frame) → stable-segment detection → sign sequence
```
**Strengths:** Leverages V1's 57-sign vocabulary; interpretable per-sign output.
**Limitation:** No learned temporal composition; depends on V1 frame-level accuracy.

### Pipeline 2: Temporal Model (end-to-end)
```
Video → MediaPipe → Landmarks (T,126) → BiLSTM/GRU + Attention → class prediction
```
**Strengths:** Learns temporal patterns directly; can model coarticulation.
**Limitation:** Needs more data than V1; no intermediate sign sequence.

---

## Files added (non-exhaustive)

```
src/config/settings.py                  (extended — additive constants only)
src/preprocessing/__init__.py           (new)
src/preprocessing/normalization.py      (new)
src/preprocessing/landmarks.py          (new)
src/preprocessing/temporal.py           (new)
src/models/continuous/__init__.py       (new)
src/models/continuous/base.py           (new)
src/models/continuous/attention.py      (new)
src/models/continuous/bilstm.py         (new)
src/models/continuous/gru.py            (new)
src/models/continuous/v1_adapter.py     (new)
src/continuous_translation/__init__.py  (new)
src/continuous_translation/pipeline.py  (new)
src/continuous_translation/inference.py (new)
scripts/gen_synthetic_landmarks.py      (new)
scripts/train_continuous.py             (new)
tests/test_temporal_preprocessing.py    (new)
tests/test_continuous_models.py         (new)
tests/test_v1_adapter.py                (new)
data/continuous/README.md               (new)
```

---

## What remains to be done

### Data (highest priority)
- [ ] Acquire real LST video corpus (Mhimdi ORTOLANG or self-recorded)
- [ ] Even 10 phrases × 5 signers × 5 repetitions would be a huge improvement
- [ ] Document camera, FPS, lighting, signer consent for any self-recorded data

### Models
- [ ] Train both BiLSTM and GRU baselines on synthetic data, report numbers
- [ ] When real data is available: retrain and compare
- [ ] Explore Transformer-based temporal model if baselines are insufficient
- [ ] Investigate ST-GCN if skeleton graph structure proves beneficial

### Linguistic layer
- [ ] Gloss / sign sequence → French phrase mapping
- [ ] Language model integration (n-gram, then small transformer)
- [ ] Evaluation with BLEU / ROUGE on phrase-level output

### Deployment
- [ ] Integrate temporal model into FastAPI backend (`api/main.py`)
- [ ] Extend WebSocket interface to support temporal inference
- [ ] Update `web/index.html` with V2 controls (model selector, mode toggle)

---

## Rules

1. Never present synthetic sequences as real LST phrases.
2. Always report synthetic and real performance separately.
3. V1 code is never modified.
4. Speaker-independent evaluation is mandatory.
5. Real data acquisition is the highest priority after this pipeline is validated.
