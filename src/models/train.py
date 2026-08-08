"""
Training script for TSL Classifier
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import logging

from src.models.classifier import TSLClassifier, create_model
from src.config.settings import (
    NUM_CLASSES, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    IMAGE_SIZE, PROCESSED_DATA_DIR, MODELS_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TSLDataset(Dataset):
    """
    Dataset for TSL classification
    """
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        transform: Optional[object] = None
    ):
        """
        Initialize dataset
        
        Args:
            features: Feature array of shape (N, 126)
            labels: Label array of shape (N,)
            transform: Optional transform
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.transform:
            feature = self.transform(feature)
        
        return feature, label


class TSLTrainer:
    """
    Trainer for TSL Classifier
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = LEARNING_RATE,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Classifier model
            device: Training device (cuda/cpu)
            learning_rate: Learning rate
            class_weights: optional tensor of per-class weights for the
                CrossEntropyLoss (helps with class imbalance)
        """
        self.model = model.to(device)
        self.device = device

        # Loss function (optionally weighted)
        if class_weights is not None:
            class_weights = class_weights.to(device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )

        # Scheduler (verbose was removed in recent torch versions)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(
        self,
        dataloader: DataLoader
    ) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            dataloader: Training dataloader
            
        Returns:
            Average loss and accuracy
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in dataloader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(
        self,
        dataloader: DataLoader
    ) -> Tuple[float, float]:
        """
        Validate model
        
        Args:
            dataloader: Validation dataloader
            
        Returns:
            Average loss and accuracy
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                total_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = NUM_EPOCHS,
        save_best: bool = True,
        early_stop_patience: int = 15,
        best_model_path: Optional[Path] = None,
    ) -> dict:
        """
        Full training loop with early stopping.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Max number of epochs
            save_best: Save the best-val-acc model
            early_stop_patience: stop if val_acc hasn't improved for this many
                epochs (0 disables early stopping)
            best_model_path: path where the best checkpoint is written

        Returns:
            Training history
        """
        best_val_acc = 0.0
        epochs_since_best = 0

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            self.scheduler.step(val_loss)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            logger.info(
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            improved = val_acc > best_val_acc
            if improved:
                best_val_acc = val_acc
                epochs_since_best = 0
                if save_best:
                    path = best_model_path or (MODELS_DIR / "best_model.pth")
                    self.save_model(path)
                    logger.info(f"Saved best model (val_acc={val_acc:.4f}) to {path}")
            else:
                epochs_since_best += 1
                if early_stop_patience > 0 and epochs_since_best >= early_stop_patience:
                    logger.info(
                        f"Early stopping after {epoch+1} epochs "
                        f"(no val_acc improvement for {early_stop_patience} epochs). "
                        f"Best val_acc={best_val_acc:.4f}"
                    )
                    break

        return self.history
    
    def save_model(self, path: Path, class_names: Optional[List[str]] = None):
        """
        Save model checkpoint.

        Persisting `class_names` (the index->label mapping) inside the
        checkpoint is critical: it guarantees the inference code uses the same
        class order that was used at training time, even if the dictionary is
        later changed.

        Args:
            path: Save path
            class_names: list of canonical TSL labels in index order
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'class_names': class_names,
            'num_classes': self.model.num_classes if hasattr(self.model, 'num_classes') else None,
        }, path)
    
    def load_model(self, path: Path):
        """
        Load model checkpoint
        
        Args:
            path: Load path
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']


def train_tsl_classifier(
    features_path: Optional[Path] = None,
    labels_path: Optional[Path] = None,
    class_names_path: Optional[Path] = None,
    num_classes: int = NUM_CLASSES,
    batch_size: int = BATCH_SIZE,
    num_epochs: int = NUM_EPOCHS,
    save_best: bool = True,
) -> dict:
    """
    Main training function.

    Args:
        features_path: Path to features .npy (defaults to processed/features.npy)
        labels_path:   Path to labels .npy   (defaults to processed/labels.npy)
        class_names_path: Path to class_names.txt (defaults to processed/class_names.txt)
        num_classes: Number of classes (must match the checkpoint's class_names)
        batch_size: Batch size
        num_epochs: Number of epochs
        save_best: Save the best-val-acc checkpoint with class_names embedded

    Returns:
        Training history
    """
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Default paths
    if features_path is None:
        features_path = PROCESSED_DATA_DIR / "features.npy"
    if labels_path is None:
        labels_path = PROCESSED_DATA_DIR / "labels.npy"
    if class_names_path is None:
        class_names_path = PROCESSED_DATA_DIR / "class_names.txt"

    # Load data
    if features_path.exists() and labels_path.exists():
        features = np.load(features_path)
        labels = np.load(labels_path)
        logger.info(f"Loaded features {features.shape}, labels {labels.shape}")
    else:
        logger.warning(
            "Processed data not found — using DUMMY data. "
            "Run `python scripts/prepare_data.py` first for real training."
        )
        features = np.random.randn(1000, 126).astype(np.float32)
        labels = np.random.randint(0, num_classes, 1000)

    # Load class names if available (single source of truth)
    class_names: Optional[List[str]] = None
    if class_names_path.exists():
        with open(class_names_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(class_names)} class names from {class_names_path}")
        # Keep num_classes consistent with class_names
        if class_names and len(class_names) != num_classes:
            logger.warning(
                f"NUM_CLASSES={num_classes} but class_names has {len(class_names)}; "
                f"using class_names length for the model."
            )
            num_classes = len(class_names)

    # Split data (stratified if possible)
    from sklearn.model_selection import train_test_split
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
    except ValueError:
        # Some classes may have too few samples for stratification
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

    # Create datasets & loaders
    train_dataset = TSLDataset(X_train, y_train)
    val_dataset = TSLDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    # Compute class weights (inverse frequency) to counter class imbalance.
    # Balanced augmentation already helps; weights give a further nudge.
    class_weights = None
    try:
        unique, counts = np.unique(labels, return_counts=True)
        weights = np.zeros(num_classes, dtype=np.float32)
        for cls, cnt in zip(unique, counts):
            if cls < num_classes:
                weights[cls] = 1.0 / max(cnt, 1)
        weights = weights / weights.mean()  # normalize so mean=1
        class_weights = torch.from_numpy(weights)
        logger.info(f"Using class weights (min={weights.min():.3f}, max={weights.max():.3f})")
    except Exception as e:
        logger.warning(f"Could not compute class weights: {e}")

    # Create model
    model = create_model("landmark", num_classes)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = TSLTrainer(model, device, class_weights=class_weights)

    # Train (the trainer.save_model call inside train() doesn't carry
    # class_names; we wrap it to persist them).
    original_save = trainer.save_model

    def save_with_class_names(path: Path):
        original_save(path, class_names=class_names)

    trainer.save_model = save_with_class_names
    best_model_path = MODELS_DIR / "best_model.pth"
    history = trainer.train(
        train_loader, val_loader,
        num_epochs=num_epochs,
        save_best=save_best,
        early_stop_patience=20,
        best_model_path=best_model_path,
    )

    # Always persist the final model too (with class_names)
    final_ckpt = MODELS_DIR / "final_model.pth"
    trainer.save_model(final_ckpt)
    logger.info(f"Saved final model to {final_ckpt}")

    return history


if __name__ == "__main__":
    # Run training
    history = train_tsl_classifier()
    print("Training completed!")
