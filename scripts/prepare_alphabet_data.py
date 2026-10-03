"""
Prepare Alphabet Data - Adapt HaGRID dataset for TSL finger spelling.

Downloads HaGRID gestures, maps them to TSL Arabic alphabet,
extracts MediaPipe hand landmarks, and saves as numpy arrays.

Usage:
    python scripts/prepare_alphabet_data.py --download
    python scripts/prepare_alphabet_data.py --prepare
    python scripts/prepare_alphabet_data.py --download --prepare
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import cv2
import json
import zipfile
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import (
    ALPHABET_RAW_DIR,
    ALPHABET_PROCESSED_DIR,
    HAGRID_DATA_DIR,
    RAW_DATA_DIR,
)
from src.models.landmark_extractor import MediaPipeLandmarkExtractor
from src.languages.alphabet import TSLAlphabet
from sklearn.model_selection import train_test_split


# Mapping: HaGRID letter -> TSL alphabet label
# HaGRID uses English letter names; we map to TSL Arabic alphabet labels
HAGRID_TO_TSL_MAP = {
    "A": "alef",
    "B": "ba",
    "C": "ta",      # Approximation: C ≈ ت
    "D": "dal",
    "E": "ha2",     # E ≈ ه
    "F": "fa",
    "G": "ghayn",   # G ≈ غ
    "H": "ha",      # H ≈ ح
    "I": "ya",      # I ≈ ي
    "J": "jim",
    "K": "kaf",
    "L": "lam",
    "M": "mim",
    "N": "nun",
    "O": "waw",     # O ≈ و
    "P": "qaf",     # P ≈ ق
    "Q": "kha",     # Q ≈ خ
    "R": "ra",
    "S": "sin",
    "T": "ta",
    "U": "ta_h",    # U ≈ ط
    "V": "dhal",    # V ≈ ذ
    "W": "sad",     # W ≈ ص
    "X": "shin",    # X ≈ ش
    "Y": "ayn",     # Y ≈ ع
    "Z": "zay",
    # Digits
    "0": "digit_0",
    "1": "digit_1",
    "2": "digit_2",
    "3": "digit_3",
    "4": "digit_4",
    "5": "digit_5",
    "6": "digit_6",
    "7": "digit_7",
    "8": "digit_8",
    "9": "digit_9",
}


def download_hagrid():
    """Download HaGRID dataset from Kaggle."""
    print("=" * 60)
    print("HaGRID Dataset Download")
    print("=" * 60)
    print()
    print("The HaGRID dataset must be downloaded manually from Kaggle:")
    print(f"  URL: https://www.kaggle.com/datasets/inktracking/hagrid")
    print()
    print(f"Place the downloaded archive in: {HAGRID_DATA_DIR}")
    print("Then extract it so that image folders are at:")
    print(f"  {HAGRID_DATA_DIR}/images/<letter>/<image>.jpg")
    print()
    print("Alternatively, you can use the Kaggle CLI:")
    print("  pip install kaggle")
    print("  kaggle datasets download -d ink_tracking/hagrid")
    print(f"  unzip hagrid.zip -d {HAGRID_DATA_DIR}")
    print()

    # Check if already downloaded
    if HAGRID_DATA_DIR.exists():
        images_dir = HAGRID_DATA_DIR / "images"
        if images_dir.exists():
            letter_dirs = [d for d in images_dir.iterdir() if d.is_dir()]
            print(f"Found {len(letter_dirs)} letter directories in {images_dir}")
            return True

    return False


def save_splits(features: np.ndarray, labels: np.ndarray):
    """Save stratified 80/10/10 train/val/test splits (random_state=42).

    Mirrors scripts/prepare_arsl.py so the X_*/y_* files in
    ALPHABET_PROCESSED_DIR are always consistent with the dataset that
    produced them (no stale splits from a different source).
    """
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
    except ValueError:
        # Too few samples per class for stratification
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

    np.save(ALPHABET_PROCESSED_DIR / "X_train.npy", X_train)
    np.save(ALPHABET_PROCESSED_DIR / "y_train.npy", y_train)
    np.save(ALPHABET_PROCESSED_DIR / "X_val.npy", X_val)
    np.save(ALPHABET_PROCESSED_DIR / "y_val.npy", y_val)
    np.save(ALPHABET_PROCESSED_DIR / "X_test.npy", X_test)
    np.save(ALPHABET_PROCESSED_DIR / "y_test.npy", y_test)
    print(f"  Splits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")


def prepare_data(max_images_per_class: int = 200, augment: bool = True):
    """
    Process HaGRID images into MediaPipe landmarks for TSL alphabet.

    Args:
        max_images_per_class: Maximum images per letter class
        augment: Apply data augmentation
    """
    print("=" * 60)
    print("Preparing TSL Alphabet Dataset from HaGRID")
    print("=" * 60)

    alphabet = TSLAlphabet()
    extractor = MediaPipeLandmarkExtractor(max_num_hands=2)

    images_dir = HAGRID_DATA_DIR / "images"
    if not images_dir.exists():
        print(f"ERROR: HaGRID images not found at {images_dir}")
        print("Please download HaGRID first: python scripts/prepare_alphabet_data.py --download")
        return

    all_features = []
    all_labels = []
    label_names = alphabet.get_all_labels()

    # Process each HaGRID letter
    for hagrid_letter, tsl_label in HAGRID_TO_TSL_MAP.items():
        if tsl_label not in label_names:
            # SPELL_LABELS has 28 letters + space/backspace/finish — no digit
            # classes, so label_names.index(tsl_label) would raise ValueError.
            print(f"  WARNING: skipping {hagrid_letter} -> {tsl_label}: "
                  "not in SPELL_LABELS (no digit classes in the TSL alphabet)")
            continue

        letter_dir = images_dir / hagrid_letter
        if not letter_dir.exists():
            print(f"  Skipping {hagrid_letter} (not found)")
            continue

        images = list(letter_dir.glob("*.jpg")) + list(letter_dir.glob("*.png"))
        if not images:
            print(f"  Skipping {hagrid_letter} (no images)")
            continue

        # Limit images per class
        images = images[:max_images_per_class]
        label_idx = label_names.index(tsl_label)

        extracted = 0
        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            features = extractor.extract_features(frame)
            if features is None:
                continue

            all_features.append(features)
            all_labels.append(label_idx)
            extracted += 1

            # Augmentation: add noisy version
            if augment and extracted < max_images_per_class:
                noise = np.random.normal(0, 0.02, features.shape).astype(np.float32)
                augmented = features + noise
                all_features.append(augmented)
                all_labels.append(label_idx)

        print(f"  {hagrid_letter} -> {tsl_label}: {extracted} images extracted")

    extractor.release()

    if not all_features:
        print("ERROR: No features extracted. Check HaGRID dataset structure.")
        return

    # Convert to numpy arrays
    features = np.array(all_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    # Shuffle
    indices = np.random.permutation(len(features))
    features = features[indices]
    labels = labels[indices]

    # Save
    ALPHABET_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ALPHABET_PROCESSED_DIR / "features.npy", features)
    np.save(ALPHABET_PROCESSED_DIR / "labels.npy", labels)

    # Save class names
    with open(ALPHABET_PROCESSED_DIR / "class_names.txt", "w", encoding="utf-8") as f:
        for name in label_names:
            f.write(name + "\n")

    # Keep the X_*/y_* splits in sync with this dataset
    save_splits(features, labels)

    # Save stats
    stats = {
        "total_samples": len(features),
        "num_classes": len(label_names),
        "class_distribution": {
            label_names[i]: int(np.sum(labels == i))
            for i in range(len(label_names))
        },
    }
    with open(ALPHABET_PROCESSED_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print()
    print(f"Dataset saved to {ALPHABET_PROCESSED_DIR}")
    print(f"  Features: {features.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Classes: {len(label_names)}")
    print(f"  Total samples: {len(features)}")


def create_synthetic_dataset():
    """
    Create a synthetic alphabet dataset for testing.
    Generates random landmark-like features for each letter class.
    Useful for verifying the pipeline works before getting real HaGRID data.
    """
    print("=" * 60)
    print("Creating Synthetic TSL Alphabet Dataset")
    print("=" * 60)

    alphabet = TSLAlphabet()
    label_names = alphabet.get_all_labels()
    num_classes = len(label_names)
    samples_per_class = 100

    np.random.seed(42)

    all_features = []
    all_labels = []

    for i, label in enumerate(label_names):
        # Generate cluster around a random center
        center = np.random.randn(126).astype(np.float32) * 0.5
        noise = np.random.randn(samples_per_class, 126).astype(np.float32) * 0.1
        features = center + noise
        all_features.append(features)
        all_labels.extend([i] * samples_per_class)
        print(f"  {label}: {samples_per_class} synthetic samples")

    features = np.vstack(all_features)
    labels = np.array(all_labels, dtype=np.int64)

    # Shuffle
    indices = np.random.permutation(len(features))
    features = features[indices]
    labels = labels[indices]

    # Save
    ALPHABET_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ALPHABET_PROCESSED_DIR / "features.npy", features)
    np.save(ALPHABET_PROCESSED_DIR / "labels.npy", labels)

    with open(ALPHABET_PROCESSED_DIR / "class_names.txt", "w", encoding="utf-8") as f:
        for name in label_names:
            f.write(name + "\n")

    # Keep the X_*/y_* splits in sync with this dataset
    save_splits(features, labels)

    print()
    print(f"Synthetic dataset saved to {ALPHABET_PROCESSED_DIR}")
    print(f"  Features: {features.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Total: {len(features)} samples across {num_classes} classes")


def main():
    parser = argparse.ArgumentParser(description="Prepare TSL alphabet dataset")
    parser.add_argument("--download", action="store_true",
                        help="Show download instructions for HaGRID")
    parser.add_argument("--prepare", action="store_true",
                        help="Process HaGRID into landmarks")
    parser.add_argument("--synthetic", action="store_true",
                        help="Create synthetic dataset for testing")
    parser.add_argument("--max-per-class", type=int, default=200,
                        help="Max images per class (default: 200)")
    parser.add_argument("--no-augment", action="store_true",
                        help="Disable data augmentation")
    args = parser.parse_args()

    if args.synthetic:
        create_synthetic_dataset()
    elif args.download or args.prepare:
        if args.download:
            download_hagrid()
        if args.prepare:
            prepare_data(
                max_images_per_class=args.max_per_class,
                augment=not args.no_augment,
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
