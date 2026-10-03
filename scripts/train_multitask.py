"""
Train Multi-Task Classifier (Version B) - Single model for signs + alphabet.

Trains the MultiTaskClassifier on both TSL signs (57 classes) and
alphabet (38 classes) with a shared backbone.

Usage:
    python scripts/train_multitask.py
    python scripts/train_multitask.py --epochs 50 --alpha 0.5 --beta 0.5
"""

import argparse
import random
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import (
    PROCESSED_DATA_DIR,
    ALPHABET_PROCESSED_DIR,
    MODELS_DIR,
    NUM_CLASSES,
    ALPHABET_NUM_CLASSES,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    EARLY_STOP_PATIENCE,
)
from src.models.multi_task_classifier import MultiTaskClassifier
from src.languages.alphabet import TSLAlphabet
from src.languages.dictionary import TSLDictionary


def _set_seed(seed: int = 42) -> None:
    """Seed python/numpy/torch RNGs so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultiTaskDataset(Dataset):
    """
    Dataset that returns the same landmark vector for both tasks.
    Sign labels come from W7CH, alphabet labels from HaGRID.
    Since they are different datasets, we train them separately
    and alternate batches.
    """

    def __init__(self, features, labels, task="sign"):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.task = task

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_sign_data():
    """Load W7CH sign dataset."""
    features_path = PROCESSED_DATA_DIR / "features.npy"
    labels_path = PROCESSED_DATA_DIR / "labels.npy"

    if not features_path.exists():
        print(f"ERROR: Sign dataset not found at {features_path}")
        print("Run: python scripts/prepare_data.py")
        sys.exit(1)

    features = np.load(features_path)
    labels = np.load(labels_path)
    print(f"Sign data: {len(features)} samples, {len(np.unique(labels))} classes")
    return features, labels


def load_spell_data():
    """Load alphabet dataset."""
    features_path = ALPHABET_PROCESSED_DIR / "features.npy"
    labels_path = ALPHABET_PROCESSED_DIR / "labels.npy"

    if not features_path.exists():
        print(f"ERROR: Alphabet dataset not found at {features_path}")
        print("Run: python scripts/prepare_alphabet_data.py --prepare")
        sys.exit(1)

    features = np.load(features_path)
    labels = np.load(labels_path)
    print(f"Spell data: {len(features)} samples, {len(np.unique(labels))} classes")
    return features, labels


def train_model(args):
    """Train the multi-task classifier."""
    _set_seed(42)
    print("=" * 60)
    print("Training Multi-Task Classifier (Version B)")
    print("=" * 60)
    print(f"  Sign classes: {NUM_CLASSES}, Spell classes: {ALPHABET_NUM_CLASSES}")
    print(f"  Alpha (sign weight): {args.alpha}, Beta (spell weight): {args.beta}")

    # Load both datasets
    sign_features, sign_labels = load_sign_data()
    spell_features, spell_labels = load_spell_data()

    # Split sign data
    X_sign_train, X_sign_temp, y_sign_train, y_sign_temp = train_test_split(
        sign_features, sign_labels, test_size=0.2, random_state=42, stratify=sign_labels
    )
    X_sign_val, X_sign_test, y_sign_val, y_sign_test = train_test_split(
        X_sign_temp, y_sign_temp, test_size=0.5, random_state=42, stratify=y_sign_temp
    )

    # Split spell data
    X_spell_train, X_spell_temp, y_spell_train, y_spell_temp = train_test_split(
        spell_features, spell_labels, test_size=0.2, random_state=42, stratify=spell_labels
    )
    X_spell_val, X_spell_test, y_spell_val, y_spell_test = train_test_split(
        X_spell_temp, y_spell_temp, test_size=0.5, random_state=42, stratify=y_spell_temp
    )

    # Create datasets
    sign_train_ds = MultiTaskDataset(X_sign_train, y_sign_train, "sign")
    sign_val_ds = MultiTaskDataset(X_sign_val, y_sign_val, "sign")
    spell_train_ds = MultiTaskDataset(X_spell_train, y_spell_train, "spell")
    spell_val_ds = MultiTaskDataset(X_spell_val, y_spell_val, "spell")

    sign_train_loader = DataLoader(sign_train_ds, batch_size=args.batch_size, shuffle=True)
    sign_val_loader = DataLoader(sign_val_ds, batch_size=args.batch_size)
    spell_train_loader = DataLoader(spell_train_ds, batch_size=args.batch_size, shuffle=True)
    spell_val_loader = DataLoader(spell_val_ds, batch_size=args.batch_size)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskClassifier(
        num_sign_classes=NUM_CLASSES,
        num_spell_classes=ALPHABET_NUM_CLASSES,
    )
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss + optimizer
    sign_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    spell_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "sign_acc": [], "spell_acc": []}

    print()
    for epoch in range(args.epochs):
        t0 = time.time()

        # Train on both tasks alternately
        model.train()
        total_train_loss = 0
        sign_batches = 0
        spell_batches = 0

        # Interleave sign and spell batches
        sign_iter = iter(sign_train_loader)
        spell_iter = iter(spell_train_loader)
        num_batches = max(len(sign_train_loader), len(spell_train_loader))

        for _ in range(num_batches):
            optimizer.zero_grad()
            total_loss = 0

            # Sign task batch
            try:
                X_sign, y_sign = next(sign_iter)
                X_sign, y_sign = X_sign.to(device), y_sign.to(device)
                sign_logits, _ = model(X_sign)
                sign_loss = sign_criterion(sign_logits, y_sign)
                total_loss += args.alpha * sign_loss
                sign_batches += 1
            except StopIteration:
                pass

            # Spell task batch
            try:
                X_spell, y_spell = next(spell_iter)
                X_spell, y_spell = X_spell.to(device), y_spell.to(device)
                _, spell_logits = model(X_spell)
                spell_loss = spell_criterion(spell_logits, y_spell)
                total_loss += args.beta * spell_loss
                spell_batches += 1
            except StopIteration:
                pass

            if total_loss > 0:
                total_loss.backward()
                optimizer.step()
                total_train_loss += total_loss.item()

        avg_train_loss = total_train_loss / max(sign_batches + spell_batches, 1)

        # Validate
        model.eval()
        val_loss = 0
        sign_correct = 0
        spell_correct = 0
        sign_total = 0
        spell_total = 0

        with torch.no_grad():
            for X_batch, y_batch in sign_val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                sign_logits, _ = model(X_batch)
                loss = sign_criterion(sign_logits, y_batch)
                val_loss += loss.item() * len(X_batch)
                preds = torch.argmax(sign_logits, dim=-1)
                sign_correct += (preds == y_batch).sum().item()
                sign_total += len(X_batch)

            for X_batch, y_batch in spell_val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                _, spell_logits = model(X_batch)
                loss = spell_criterion(spell_logits, y_batch)
                val_loss += loss.item() * len(X_batch)
                preds = torch.argmax(spell_logits, dim=-1)
                spell_correct += (preds == y_batch).sum().item()
                spell_total += len(X_batch)

        val_loss /= (sign_total + spell_total)
        sign_acc = sign_correct / max(sign_total, 1)
        spell_acc = spell_correct / max(spell_total, 1)

        scheduler.step(val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["sign_acc"].append(sign_acc)
        history["spell_acc"].append(spell_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Loss: {avg_train_loss:.4f}/{val_loss:.4f} | "
            f"Sign: {sign_acc:.4f} | "
            f"Spell: {spell_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, history, "best")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    save_checkpoint(model, optimizer, epoch, history, "final")
    print("\nTraining complete.")


def save_checkpoint(model, optimizer, epoch, history, suffix):
    """Save model checkpoint."""
    path = MODELS_DIR / f"multitask_{suffix}.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "history": history,
        "sign_class_names": TSLDictionary().get_all_labels(),
        "spell_class_names": TSLAlphabet().get_all_labels(),
    }, path)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train Multi-Task Classifier")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Weight for sign task loss")
    parser.add_argument("--beta", type=float, default=0.5,
                        help="Weight for spell task loss")
    args = parser.parse_args()

    train_model(args)


if __name__ == "__main__":
    main()
