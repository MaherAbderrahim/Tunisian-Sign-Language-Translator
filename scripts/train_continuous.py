#!/usr/bin/env python3
"""Train BiLSTM / GRU temporal baselines on the synthetic mini-benchmark.

Usage:
    python scripts/train_continuous.py --model bilstm
    python scripts/train_continuous.py --model gru
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.settings import (
    CONTINUOUS_DATA_DIR,
    CONTINUOUS_MODELS_DIR,
    TEMPORAL_INPUT_DIM,
    TEMPORAL_HIDDEN_DIM,
    TEMPORAL_NUM_LAYERS,
    TEMPORAL_DROPOUT,
    CONTINUOUS_BATCH_SIZE,
    CONTINUOUS_LEARNING_RATE,
    CONTINUOUS_NUM_EPOCHS,
    CONTINUOUS_EARLY_STOP_PATIENCE,
)
from src.models.continuous.bilstm import BiLSTMAttentionModel
from src.models.continuous.gru import GRUAttentionModel
from src.preprocessing.temporal import pad_sequence_2d, collate_padded


class SyntheticSignDataset(Dataset):
    def __init__(self, split: str, landmarks_dir: Path, metadata_path: Path):
        self.sequences: list[np.ndarray] = []
        self.labels: list[int] = []
        self.max_len = 0
        with open(metadata_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] != split:
                    continue
                path = landmarks_dir / f"{row['sample_id']}.npy"
                if not path.exists():
                    continue
                seq = np.load(path)
                self.sequences.append(seq)
                self.labels.append(int(row["label_idx"]))
                if seq.shape[0] > self.max_len:
                    self.max_len = seq.shape[0]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X, M, Y in loader:
        X, M, Y = X.to(device), M.to(device), Y.to(device)
        optimizer.zero_grad()
        logits = model(X, mask=M)
        loss = criterion(logits, Y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * len(Y)
        correct += (logits.argmax(-1) == Y).sum().item()
        total += len(Y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    for X, M, Y in loader:
        X, M, Y = X.to(device), M.to(device), Y.to(device)
        logits = model(X, mask=M)
        loss = criterion(logits, Y)
        total_loss += loss.item() * len(Y)
        preds = logits.argmax(-1)
        correct += (preds == Y).sum().item()
        total += len(Y)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(Y.cpu().numpy().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


def compute_f1_macro(preds, labels, num_classes):
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)
    for p, l in zip(preds, labels):
        if p == l:
            tp[l] += 1
        else:
            fp[p] += 1
            fn[l] += 1
    precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    return float(f1.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bilstm", "gru"], default="bilstm")
    parser.add_argument("--epochs", type=int, default=CONTINUOUS_NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=CONTINUOUS_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=CONTINUOUS_BATCH_SIZE)
    parser.add_argument("--hidden", type=int, default=TEMPORAL_HIDDEN_DIM)
    parser.add_argument("--layers", type=int, default=TEMPORAL_NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=TEMPORAL_DROPOUT)
    args = parser.parse_args()

    landmarks_dir = CONTINUOUS_DATA_DIR / "landmarks"
    meta_path = CONTINUOUS_DATA_DIR / "metadata.csv"
    if not meta_path.exists():
        print("ERROR: metadata.csv not found. Run gen_synthetic_landmarks.py first.")
        sys.exit(1)

    train_ds = SyntheticSignDataset("train", landmarks_dir, meta_path)
    val_ds = SyntheticSignDataset("val", landmarks_dir, meta_path)
    test_ds = SyntheticSignDataset("test", landmarks_dir, meta_path)

    if len(train_ds) == 0:
        print("ERROR: No training data found.")
        sys.exit(1)

    max_len = max(train_ds.max_len, val_ds.max_len, test_ds.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_padded(b, max_len=max_len))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=lambda b: collate_padded(b, max_len=max_len))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=lambda b: collate_padded(b, max_len=max_len))

    num_classes = len(set(train_ds.labels) | set(val_ds.labels) | set(test_ds.labels))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ModelClass = BiLSTMAttentionModel if args.model == "bilstm" else GRUAttentionModel
    model = ModelClass(
        num_classes=num_classes,
        input_dim=TEMPORAL_INPUT_DIM,
        hidden_dim=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)

    print(f"Model: {args.model} | Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} | MaxLen: {max_len}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc, patience_counter = 0.0, 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | Train {train_loss:.4f}/{train_acc:.3f} | "
              f"Val {val_loss:.4f}/{val_acc:.3f} | {dt:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= CONTINUOUS_EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    test_f1 = compute_f1_macro(test_preds, test_labels, num_classes)

    from collections import Counter
    label_counts = Counter(test_labels)
    print(f"\n--- Test Results ({args.model}) ---")
    print(f"  Accuracy:      {test_acc:.4f}")
    print(f"  Macro F1:      {test_f1:.4f}")
    print(f"  Test samples:  {len(test_ds)}")
    print(f"  Class dist:    {dict(sorted(label_counts.items()))}")

    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_type": args.model,
        "num_classes": num_classes,
        "input_dim": TEMPORAL_INPUT_DIM,
        "hidden_dim": args.hidden,
        "num_layers": args.layers,
        "max_len": max_len,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "class_names": [str(i) for i in range(num_classes)],
    }
    out_path = CONTINUOUS_MODELS_DIR / f"synthetic_{args.model}_best.pth"
    torch.save(ckpt, out_path)
    print(f"  Saved to: {out_path}")


if __name__ == "__main__":
    main()
