"""Training loop, evaluation and metrics (shared by all experiments)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .paths import RESULTS_DIR, SEED


class Trainer:
    """Small trainer with ReduceLROnPlateau + early stopping on val accuracy."""

    def __init__(self, model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-4,
                 label_smoothing: float = 0.1, patience: int = 20,
                 seed: int = SEED):
        from .models import set_seed
        set_seed(seed)
        self.model = model
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                          weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5)
        self.patience = patience
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [],
                        "val_acc": []}
        self.best_val_acc = 0.0
        self.best_state = None
        self.epochs_run = 0

    def _run_epoch(self, loader, train: bool):
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for x, y in loader:
                logits = self.model(x)
                loss = self.criterion(logits, y)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item() * y.size(0)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        return total_loss / total, correct / total

    def fit(self, train_loader, val_loader, max_epochs: int = 100,
            early_stop: bool = True):
        stall = 0
        for epoch in range(max_epochs):
            t0 = time.perf_counter()
            tr_loss, tr_acc = self._run_epoch(train_loader, True)
            va_loss, va_acc = self._run_epoch(val_loader, False)
            self.scheduler.step(va_loss)
            self.history["train_loss"].append(tr_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_loss"].append(va_loss)
            self.history["val_acc"].append(va_acc)
            self.epochs_run += 1
            if va_acc > self.best_val_acc:
                self.best_val_acc = va_acc
                self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                stall = 0
            else:
                stall += 1
            dt = time.perf_counter() - t0
            print(f"  ep {epoch + 1}/{max_epochs} "
                  f"train {tr_loss:.4f}/{tr_acc:.4f} val {va_loss:.4f}/{va_acc:.4f} "
                  f"({dt:.1f}s) best {self.best_val_acc:.4f}")
            if early_stop and stall >= self.patience:
                print(f"  early stop @ {epoch + 1} (best val acc {self.best_val_acc:.4f})")
                break
        return self.history

    def restore_best(self):
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)


def predict(model: nn.Module, loader) -> tuple:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            ys.append(y.numpy())
            ps.append(logits.argmax(1).numpy())
    return np.concatenate(ps), np.concatenate(ys)


def predict_on_features(model: nn.Module, features: np.ndarray,
                        batch_size: int = 256) -> np.ndarray:
    """Batch inference on a raw feature matrix (for robustness tests)."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            x = torch.from_numpy(features[i:i + batch_size]).float()
            outs.append(model(x).argmax(1).numpy())
    return np.concatenate(outs)


def full_metrics(y_true, y_pred, num_classes: int = 57) -> dict:
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 confusion_matrix)
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0)
    present = sup > 0
    macro_f1 = float(f1[present].mean())
    macro_prec = float(prec[present].mean())
    macro_rec = float(rec[present].mean())
    wprec, wrec, wf1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return {
        "accuracy": acc,
        "precision": float(wprec),
        "recall": float(wrec),
        "f1": float(wf1),
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "per_class_precision": prec.tolist(),
        "per_class_recall": rec.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": sup.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def save_metrics(name: str, metrics: dict, extra: dict = None):
    payload = {"model": name, "metrics": metrics}
    if extra:
        payload["extra"] = extra
    out = RESULTS_DIR / f"metrics_{name}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"metrics saved -> {out}")


def plot_training_curves(history: dict, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title(f"{title} — loss")
    axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_title(f"{title} — accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, class_names, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=5)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=5)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Truth")
    ax.set_title(title)
    fig.colorbar(im, label="rate")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_torch_benchmark(model: nn.Module, sample: torch.Tensor,
                        iters: int = 300) -> float:
    """Mean inference latency (ms) for a single sample."""
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            model(sample)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(sample)
        dt = (time.perf_counter() - t0) / iters * 1000
    return float(dt)