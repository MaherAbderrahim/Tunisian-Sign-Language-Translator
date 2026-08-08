"""
TSL Classifier for Tunisian Sign Language Recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional, Tuple
import numpy as np


class TSLClassifier(nn.Module):
    """
    Classifieur pour la Langue des Signes Tunisienne (TSL)
    
    Architecture:
    - MLP profond avec résiduels (126 -> 512 -> 256 -> 128 -> 64 -> 57)
    - BatchNorm + ReLU + Dropout
    - Init Kaiming
    - Input: Landmarks MediaPipe (126 features = 2 mains × 21 points × 3 coordonnées)
    - Output: 57 classes TSL
    """
    
    def __init__(
        self, 
        num_classes: int = 57,
        input_size: int = 126,
        dropout_rate: float = 0.2
    ):
        """
        Initialize TSL Classifier (deeper residual MLP).

        Args:
            num_classes: Number of TSL classes (57)
            input_size: Input feature size (126 for 2 hands)
            dropout_rate: Dropout rate for regularization
        """
        super(TSLClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Input block
        self.input_block = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Residual block 1: 512 -> 512
        self.res1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
        )
        
        # Residual block 2: 512 -> 256
        self.down1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Residual block 3: 256 -> 256
        self.res2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
        )
        
        # Downsample to 128
        self.down2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Final classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(64, num_classes)
        )
        
        # Initialisation des poids
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.

        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Input block
        h = self.input_block(x)
        
        # Residual block 1 + skip
        h = h + self.res1(h)
        h = nn.functional.relu(h)
        
        # Downsample to 256
        h = self.down1(h)
        
        # Residual block 2 + skip
        h = h + self.res2(h)
        h = nn.functional.relu(h)
        
        # Downsample to 128
        h = self.down2(h)
        
        # Classifier
        logits = self.classifier(h)
        return logits
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class indices
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=-1)
        return predictions
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities
        
        Args:
            x: Input tensor
            
        Returns:
            Class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=-1)
        return probabilities
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get feature embeddings
        
        Args:
            x: Input tensor
            
        Returns:
            Feature embeddings
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.landmark_encoder(x)
        return embeddings


class TSLClassifierCNN(nn.Module):
    """
    Alternative CNN-based classifier for image input
    Uses EfficientNet-B0 backbone with transfer learning
    """
    
    def __init__(
        self,
        num_classes: int = 57,
        pretrained: bool = True
    ):
        """
        Initialize CNN Classifier
        
        Args:
            num_classes: Number of TSL classes
            pretrained: Use pretrained EfficientNet weights
        """
        super(TSLClassifierCNN, self).__init__()
        
        # Load pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        )
        
        # Get the number of features from the backbone
        num_features = self.backbone.classifier[1].in_features
        
        # Replace the classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.backbone(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class labels"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=-1)
        return predictions


class EnsembleClassifier(nn.Module):
    """
    Ensemble classifier combining landmark and image features
    """
    
    def __init__(
        self,
        num_classes: int = 57,
        landmark_size: int = 126,
        dropout_rate: float = 0.3
    ):
        """
        Initialize Ensemble Classifier
        
        Args:
            num_classes: Number of TSL classes
            landmark_size: Landmark feature size
            dropout_rate: Dropout rate
        """
        super(EnsembleClassifier, self).__init__()
        
        # Landmark branch
        self.landmark_branch = nn.Sequential(
            nn.Linear(landmark_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Image branch (from EfficientNet)
        self.image_branch = nn.Sequential(
            nn.Linear(1280, 128),  # EfficientNet-B0 output
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(
        self, 
        landmarks: torch.Tensor,
        image_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with both modalities
        
        Args:
            landmarks: Landmark features
            image_features: Image features from backbone
            
        Returns:
            Logits
        """
        # Process each branch
        landmark_out = self.landmark_branch(landmarks)
        image_out = self.image_branch(image_features)
        
        # Concatenate
        combined = torch.cat([landmark_out, image_out], dim=-1)
        
        # Fusion
        logits = self.fusion(combined)
        
        return logits


def create_model(
    model_type: str = "landmark",
    num_classes: int = 57,
    pretrained: bool = True
) -> nn.Module:
    """
    Factory function to create classifier
    
    Args:
        model_type: Type of model ("landmark", "cnn", "ensemble")
        num_classes: Number of classes
        pretrained: Use pretrained weights
        
    Returns:
        Classifier model
    """
    if model_type == "landmark":
        return TSLClassifier(num_classes=num_classes)
    elif model_type == "cnn":
        return TSLClassifierCNN(num_classes=num_classes, pretrained=pretrained)
    elif model_type == "ensemble":
        return EnsembleClassifier(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
