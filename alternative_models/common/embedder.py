"""Wrapper around the pretrained MediaPipe gesture-embedding model.

The gesture embedder (Uboweja et al., 2023 — arXiv:2309.10858) is a TFLite
model trained on HaGRID (~30k+ hand images) that maps a single hand's
landmarks to a 128-dim embedding. Inputs (exactly as produced by MediaPipe's
HandLandmarker):

    hand:        (1, 21, 3) image-normalized landmarks (x, y in [0,1], z depth)
    handedness:  (1, 1) handedness score
    world_hand:  (1, 21, 3) world landmarks (meters)

We run it with the LiteRT interpreter (ai-edge-litert) and freeze it: only
the classification head on top of the 2x128-dim embedding is trained, which
mirrors Google's own Model Maker pipeline.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .paths import GESTURE_EMBEDDER_PATH

EMBEDDING_DIM = 128


class GestureEmbedder:
    """Frozen pretrained embedder, one hand at a time (TFLite)."""

    def __init__(self, model_path=None):
        from ai_edge_litert.interpreter import Interpreter
        self.interp = Interpreter(model_path=str(model_path or GESTURE_EMBEDDER_PATH))
        self.interp.allocate_tensors()
        self.inputs = self.interp.get_input_details()
        self.outputs = self.interp.get_output_details()
        names = [i["name"] for i in self.inputs]
        assert "hand" in names and "world_hand" in names and "handedness" in names, names

    def embed_hand(self, hand: np.ndarray, world: np.ndarray,
                   handedness: float) -> np.ndarray:
        """hand/world: (21, 3) float32. Returns (128,) embedding."""
        self.interp.set_tensor(self.inputs[0]["index"],
                               hand.astype(np.float32)[None])
        self.interp.set_tensor(self.inputs[1]["index"],
                               np.array([[handedness]], dtype=np.float32))
        self.interp.set_tensor(self.inputs[2]["index"],
                               world.astype(np.float32)[None])
        self.interp.invoke()
        return self.interp.get_tensor(self.outputs[0]["index"])[0].copy()

    def embed_batch(self, img_lm: np.ndarray, world_lm: np.ndarray,
                    handedness: np.ndarray, n_hands: np.ndarray,
                    batch_size: int = 64) -> np.ndarray:
        """Embed all hands. img_lm: (N, 2, 21, 3) image-normalized landmarks,
        world_lm: (N, 2, 21, 3), handedness: (N, 2), n_hands: (N, 2) mask.

        Returns (N, 256) = concat(embed(hand0), embed(hand1)); absent hands
        yield a zero embedding.
        """
        N = img_lm.shape[0]
        out = np.zeros((N, 2, EMBEDDING_DIM), dtype=np.float32)
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            for h in range(2):
                for i in range(start, end):
                    if n_hands[i, h] > 0:
                        out[i, h] = self.embed_hand(
                            img_lm[i, h], world_lm[i, h], handedness[i, h])
        return out.reshape(N, 2 * EMBEDDING_DIM)

    def latency_ms(self, hand: np.ndarray, world: np.ndarray,
                   iters: int = 200) -> float:
        import time
        h = hand.astype(np.float32)
        w = world.astype(np.float32)
        for _ in range(10):
            self.embed_hand(h, w, 0.9)
        t0 = time.perf_counter()
        for _ in range(iters):
            self.embed_hand(h, w, 0.9)
        return float((time.perf_counter() - t0) / iters * 1000)


class EmbedderHeadMLP(nn.Module):
    """Small MLP head trained on top of the frozen 256-dim embedding."""

    def __init__(self, num_classes: int = 57, input_dim: int = 256,
                 hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def extract_embeddings(dataset: dict, embedder: GestureEmbedder,
                       cache_path=None) -> dict:
    """Compute embeddings for train/val/test splits (cached to disk)."""
    from pathlib import Path
    if cache_path is not None and Path(cache_path).exists():
        z = np.load(cache_path, allow_pickle=True)
        return {k: {"embeddings": z[f"{k}_emb"], "labels": z[f"{k}_labels"]}
                for k in ["train", "val", "test"]}
    out = {}
    for split, d in dataset.items():
        emb = embedder.embed_batch(d["img_landmarks"], d["world_landmarks"],
                                   d["handedness"], d["n_hands"])
        out[split] = {"embeddings": emb, "labels": d["labels"]}
        print(f"  {split}: embeddings {emb.shape}")
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path,
                 train_emb=out["train"]["embeddings"], train_labels=out["train"]["labels"],
                 val_emb=out["val"]["embeddings"], val_labels=out["val"]["labels"],
                 test_emb=out["test"]["embeddings"], test_labels=out["test"]["labels"])
    return out