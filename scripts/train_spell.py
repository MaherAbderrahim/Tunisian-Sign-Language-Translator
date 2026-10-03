"""
Train Spell Classifier (Version A) - Separate model for finger spelling.

Trains the SpellClassifier on TSL alphabet data (38 classes).

Usage:
    python scripts/train_spell.py
    python scripts/train_spell.py --epochs 50 --lr 0.001
    python scripts/train_spell.py --synthetic  # train on synthetic data first
"""

import argparse
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import (
    ALPHABET_PROCESSED_DIR,
    ALPHABET_MODELS_DIR,
    ALPHABET_NUM_CLASSES,
    SPELL_BATCH_SIZE,
    SPELL_LEARNING_RATE,
    SPELL_NUM_EPOCHS,
    SPELL_EARLY_STOP_PATIENCE,
    SPELL_LABEL_SMOOTHING,
)
from src.models.spell_classifier import SpellClassifier
from src.languages.alphabet import TSLAlphabet


def load_data():
    """Load processed alphabet dataset."""
    features_path = ALPHABET_PROCESSED_DIR / "features.npy"
    labels_path = ALPHABET_PROCESSED_DIR / "labels.npy"

    if not features_path.exists():
        print(f"ERROR: Dataset not found at {features_path}")
        print("Run: python scripts/prepare_alphabet_data.py --prepare")
        sys.exit(1)

    features = np.load(features_path)
    labels = np.load(labels_path)

    print(f"Loaded {len(features)} samples, {len(np.unique(labels))} classes")
    return features, labels


def train_model(args):
    """Train the spell classifier."""
    print("=" * 60)
    print("Training Spell Classifier (Version A)")
    print("=" * 60)

    # Load data
    features, labels = load_data()

    # Split: 80/10/10 stratified
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Data augmentation for training
    if not args.no_augment:
        X_augmented, y_augmented = augment_data(X_train, y_train)
        X_train = np.vstack([X_train, X_augmented])
        y_train = np.concatenate([y_train, y_augmented])
        print(f"After augmentation: {len(X_train)} training samples")

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.LongTensor(y_val)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    # DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpellClassifier(num_classes=ALPHABET_NUM_CLASSES)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss + optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    print()
    for epoch in range(args.epochs):
        t0 = time.time()

        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(X_train)

        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(X_batch)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == y_batch).sum().item()
        val_loss /= len(X_val)
        val_acc = correct / len(X_val)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            save_checkpoint(model, optimizer, epoch, history, "best")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    # Final save
    save_checkpoint(model, optimizer, epoch, history, "final")

    # Test evaluation
    print()
    print("=" * 60)
    print("Test Evaluation")
    print("=" * 60)
    evaluate_model(model, X_test_t, y_test_t, device)


def augment_data(features, labels, factor=2):
    """Simple data augmentation: add Gaussian noise."""
    augmented_features = []
    augmented_labels = []
    for _ in range(factor - 1):
        noise = np.random.normal(0, 0.02, features.shape).astype(np.float32)
        augmented_features.append(features + noise)
        augmented_labels.append(labels)
    return np.vstack(augmented_features), np.concatenate(augmented_labels)


def save_checkpoint(model, optimizer, epoch, history, suffix):
    """Save model checkpoint."""
    ALPHABET_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = ALPHABET_MODELS_DIR / f"spell_classifier_{suffix}.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "history": history,
        "class_names": TSLAlphabet().get_all_labels(),
    }, path)
    print(f"  Saved: {path}")


def evaluate_model(model, X_test, y_test, device):
    """Evaluate model on test set."""
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        logits = model(X_test)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()

    y_true = y_test.numpy()
    alphabet = TSLAlphabet()
    target_names = alphabet.get_all_labels()

    print(classification_report(y_true, preds, target_names=target_names, zero_division=0))


def main():
    parser = argparse.ArgumentParser(description="Train Spell Classifier")
    parser.add_argument("--epochs", type=int, default=SPELL_NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=SPELL_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=SPELL_BATCH_SIZE)
    parser.add_argument("--patience", type=int, default=SPELL_EARLY_STOP_PATIENCE)
    parser.add_argument("--label-smoothing", type=float, default=SPELL_LABEL_SMOOTHING)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--synthetic", action="store_true",
                        help="Train on synthetic data (for pipeline testing)")
    args = parser.parse_args()

    if args.synthetic:
        from scripts.prepare_alphabet_data import create_synthetic_dataset
        create_synthetic_dataset()

    train_model(args)


if __name__ == "__main__":
    main()
