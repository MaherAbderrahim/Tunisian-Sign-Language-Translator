"""Robustness tests: the models are evaluated under perturbations of the
landmarks (noise, rotation, translation, scale) applied in image space
BEFORE wrist-normalization, so they represent realistic camera variations.

`apply_perturbation` transforms the raw image-normalized landmarks; the
normalized features are then recomputed exactly as during training.
"""
from __future__ import annotations

import numpy as np

from .data import wrist_normalize, flatten


def perturb_rotation(raw: np.ndarray, angle_deg: float,
                     rng: np.random.Generator) -> np.ndarray:
    out = raw.copy()
    for h in range(raw.shape[-3]):
        hand = out[..., h, :, :]
        valid = np.abs(hand).sum(-1) > 1e-8
        if not valid.any():
            continue
        a = float(np.deg2rad(angle_deg))
        c, s = float(np.cos(a)), float(np.sin(a))
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        xy = hand[..., :2]
        w = xy[..., 0, :].copy()  # per-sample wrist
        xy = (xy - w[..., None, :]) @ R.T + w[..., None, :]
        out[..., h, :, :2] = xy
    return out


def perturb_translation(raw: np.ndarray, shift: float,
                        rng: np.random.Generator) -> np.ndarray:
    out = raw.copy()
    n = raw.shape[0]
    dx = rng.uniform(-shift, shift, size=(n, 1)).astype(np.float32)
    dy = rng.uniform(-shift, shift, size=(n, 1)).astype(np.float32)
    out[..., :2] += np.stack([dx, dy], axis=-1)[..., None, :]
    return out


def perturb_scale(raw: np.ndarray, scale: float,
                  rng: np.random.Generator) -> np.ndarray:
    out = raw.copy()
    for h in range(raw.shape[-3]):
        hand = out[..., h, :, :]
        valid = np.abs(hand).sum(-1) > 1e-8
        if not valid.any():
            continue
        w = hand[..., 0:1, :].copy()
        out[..., h, :, :] = (hand - w) * scale + w
    return out


def perturb_noise(raw: np.ndarray, std: float,
                  rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, std, size=raw.shape).astype(np.float32)
    out = raw + noise
    for h in range(raw.shape[-3]):
        valid = np.abs(out[..., h, :, :]).sum(-1) > 1e-8
        out[..., h, :, :] = np.where(~valid[..., None], 0.0, out[..., h, :, :])
    return out


def evaluate_perturbed_model(model, raw_test: np.ndarray, y_test: np.ndarray,
                             perturb_fn, forward_fn, num_trials: int = 5,
                             batch: int = 512):
    """Perturb raw landmarks in image space, then predict with `forward_fn`
    (a callable model(raw) -> predictions). Returns (mean_acc, std_acc)."""
    rng = np.random.default_rng(0)
    accs = []
    for _ in range(num_trials):
        raw_p = perturb_fn(raw_test, rng)
        preds = forward_fn(model, raw_p, batch_size=batch)
        accs.append(float((preds == y_test).mean()))
    return float(np.mean(accs)), float(np.std(accs))


def robustness_summary(model, raw_test: np.ndarray, y_test: np.ndarray,
                       base_acc: float, forward_fn=None) -> dict:
    """Run the full robustness battery on one model. Returns dict of results.

    `forward_fn(model, raw, batch_size) -> preds`; when None the model is
    assumed to take wrist-normalized flat features directly.
    """
    from .train import predict_on_features
    if forward_fn is None:
        def forward_fn(model, raw, batch_size):
            feats = flatten(wrist_normalize(raw))
            return predict_on_features(model, feats, batch_size=batch_size)

    out = {"base_accuracy": base_acc}
    tests = {
        "noise_0.01": lambda r, g: perturb_noise(r, 0.01, g),
        "noise_0.02": lambda r, g: perturb_noise(r, 0.02, g),
        "noise_0.05": lambda r, g: perturb_noise(r, 0.05, g),
        "rotation_5": lambda r, g: perturb_rotation(r, 5.0, g),
        "rotation_10": lambda r, g: perturb_rotation(r, 10.0, g),
        "rotation_20": lambda r, g: perturb_rotation(r, 20.0, g),
        "translation_0.05": lambda r, g: perturb_translation(r, 0.05, g),
        "translation_0.10": lambda r, g: perturb_translation(r, 0.10, g),
        "scale_0.90": lambda r, g: perturb_scale(r, 0.90, g),
        "scale_1.10": lambda r, g: perturb_scale(r, 1.10, g),
    }
    for name, fn in tests.items():
        mean, std = evaluate_perturbed_model(
            model, raw_test, y_test, fn, forward_fn=forward_fn)
        out[name] = {"accuracy": mean, "std": std,
                     "drop": base_acc - mean}
        print(f"  {name}: acc={mean:.4f} (drop={base_acc - mean:+.4f})")
    return out