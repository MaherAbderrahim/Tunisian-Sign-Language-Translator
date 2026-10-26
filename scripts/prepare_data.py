"""
Script to prepare TSL dataset for training.

Real W7CH layout after `scripts/download_data.py`:
    data/raw/tsl_dataset/data/Data/<Category>/<class>/<image>.jpg

This script:
  - Walks the nested category/class structure
  - Maps dataset folder names (incl. the "métro" mojibake) to canonical TSL
    labels via TSLDictionary
  - Extracts MediaPipe landmarks per image
  - Splits the ORIGINALS 80/20 (train/val) and augments ONLY the train fold
  - Saves features.npy / labels.npy / class_names.txt (unaugmented originals)
    plus X_train.npy / y_train.npy / X_val.npy / y_val.npy (augmented train,
    held-out val)

`class_names.txt` is the single source of truth for the class-index mapping
that MUST be persisted into the trained checkpoint (see `train.py`).
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import logging

# Ensure project root is importable when run as `python scripts/prepare_data.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import (
    DATASET_ROOT, PROCESSED_DATA_DIR, IMAGE_SIZE,
)
from src.models.landmark_extractor import MediaPipeLandmarkExtractor
from src.languages.dictionary import TSLDictionary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_tsl_dataset(
    dataset_path: Optional[Path] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    """
    Load TSL dataset from `data/raw/tsl_dataset/data/Data/<Cat>/<class>/`.

    Returns:
        Tuple of (features, labels, class_names) or (None, None, None) on failure.
        class_names is the canonical label order used by the dictionary and the
        model — `labels[i]` is the index of the class in `class_names`.
    """
    if dataset_path is None:
        dataset_path = DATASET_ROOT

    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        logger.info("Please run 'python scripts/download_data.py' first")
        return None, None, None

    logger.info(f"Loading TSL dataset from {dataset_path} ...")

    # Single source of truth for class order
    dictionary = TSLDictionary()
    class_names = dictionary.get_all_labels()
    class_to_index = {name: i for i, name in enumerate(class_names)}

    extractor = MediaPipeLandmarkExtractor()

    features_list: List[np.ndarray] = []
    labels_list: List[int] = []

    seen_classes: set = set()
    skipped_classes: List[Tuple[str, str]] = []  # (folder_name, category)

    # Walk: <dataset>/Data/<Category>/<class>/<img>.jpg (handle both with and
    # without an extra "Data" parent, to be robust to layout variations).
    category_dirs = sorted(
        d for d in dataset_path.iterdir() if d.is_dir()
    )

    # If there's only a single "Data" subfolder, descend into it.
    if len(category_dirs) == 1 and category_dirs[0].name.lower() == "data":
        dataset_path = category_dirs[0]
        category_dirs = sorted(
            d for d in dataset_path.iterdir() if d.is_dir()
        )

    total_images = 0
    for category_dir in category_dirs:
        logger.info(f"Processing category: {category_dir.name}")
        for class_dir in sorted(category_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            folder_name = class_dir.name
            canonical = dictionary.canonical_label(folder_name)
            if canonical is None:
                skipped_classes.append((folder_name, category_dir.name))
                continue

            label_idx = class_to_index[canonical]
            seen_classes.add(canonical)

            # W7CH stores images as .jpg (and a few .jpeg/.png). Support all.
            image_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
            img_paths: List[Path] = []
            for ext in image_exts:
                img_paths.extend(class_dir.glob(ext))

            n_ok = 0
            for img_path in img_paths:
                try:
                    frame = cv2.imread(str(img_path))
                    if frame is None:
                        continue
                    # Resize to a consistent shape (landmarks are scale-invariant
                    # after normalization, but keeping a canonical size avoids
                    # surprises).
                    frame = cv2.resize(frame, IMAGE_SIZE)

                    features = extractor.extract_features(frame)
                    if features is None:
                        # No hand detected — skip silently
                        continue

                    features_list.append(features)
                    labels_list.append(label_idx)
                    n_ok += 1
                    total_images += 1
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")

            logger.info(
                f"  {folder_name} -> {canonical}: {n_ok}/{len(img_paths)} images"
            )

    extractor.release()

    if skipped_classes:
        logger.warning(
            f"Skipped {len(skipped_classes)} unknown dataset folders: "
            f"{[n for n, _ in skipped_classes]}"
        )

    missing_classes = set(class_names) - seen_classes
    if missing_classes:
        logger.warning(
            f"{len(missing_classes)} dictionary classes had NO matching dataset "
            f"folder: {sorted(missing_classes)}"
        )

    if not features_list:
        logger.error("No features extracted — is MediaPipe installed and the dataset present?")
        return None, None, None

    features = np.array(features_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int64)

    # Sanity: per-class counts
    unique, counts = np.unique(labels, return_counts=True)
    logger.info(
        f"Loaded {len(features)} samples across {len(unique)} classes "
        f"(dictionary has {len(class_names)} classes)."
    )
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Labels shape:   {labels.shape}")
    logger.info(
        f"Min/max per-class samples: {counts.min()} / {counts.max()}"
    )

    return features, labels, class_names


def save_processed_data(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    output_dir: Optional[Path] = None,
):
    """
    Save processed features, labels, and class names (canonical order).

    Args:
        features: (N, 126) float32 array
        labels:   (N,) int64 array of class indices
        class_names: list of canonical TSL labels in index order
        output_dir: destination directory (defaults to PROCESSED_DATA_DIR)
    """
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "labels.npy", labels)

    # Persist the class-name order — this is the contract the checkpoint must
    # also save and the inference code must use.
    with open(output_dir / "class_names.txt", "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")

    logger.info(f"Saved processed data to {output_dir}")


def _augment_landmarks_sample(
    features: np.ndarray,
    rng: np.random.default_rng,
    angle_max: float = 20.0,
    scale_range: tuple = (0.85, 1.15),
    shift_max: float = 0.1,
    jitter_std: float = 0.02,
) -> np.ndarray:
    """Apply geometric + noise augmentation to a single landmark sample.

    The (126,) vector is reshaped as (2 hands, 21 landmarks, 3 coords) and:
      - rotate (x,y) by a random angle in [-angle_max, +angle_max] degrees
      - scale by a random factor in scale_range
      - shift (x,y) by a random offset in [-shift_max, +shift_max]
      - add small Gaussian noise (jitter_std) for robustness

    z-coordinate is left unchanged (z is depth, not affected by 2D transforms).
    Returns a new (126,) float32 array.
    """
    f = features.reshape(2, 21, 3).copy().astype(np.float32)

    angle = float(rng.uniform(-angle_max, angle_max) * np.pi / 180.0)
    cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)

    scale = float(rng.uniform(*scale_range))
    shift_x = float(rng.uniform(-shift_max, shift_max))
    shift_y = float(rng.uniform(-shift_max, shift_max))

    for h in range(f.shape[0]):
        xy = f[h, :, :2]  # (21, 2)
        xy = (xy @ rot.T) * scale
        xy[:, 0] += shift_x
        xy[:, 1] += shift_y
        f[h, :, :2] = xy

    noise = rng.normal(0.0, jitter_std, f.shape).astype(np.float32)
    return (f + noise).flatten()


def create_augmented_data(
    features: np.ndarray,
    labels: np.ndarray,
    augmentation_factor: int = 2,
    target_per_class: Optional[int] = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create augmented data with per-class balancing.

    For each class:
      - keep all original samples
      - if the class has fewer than `target_per_class` samples, generate
        augmented copies (geometric + noise) until reaching the target
      - otherwise, generate `augmentation_factor - 1` augmented copies

    Args:
        features: (N, 126) original features
        labels:   (N,) original labels
        augmentation_factor: multiplier for well-populated classes
        target_per_class: minimum number of (orig+aug) samples per class;
            None disables balancing

    Returns:
        Augmented features and labels.
    """
    rng = np.random.default_rng(seed=42)
    out_feats: List[np.ndarray] = []
    out_labels: List[int] = []

    unique_labels = np.unique(labels)
    for cls in unique_labels:
        mask = labels == cls
        cls_feats = features[mask]
        n_orig = len(cls_feats)
        out_feats.append(cls_feats)
        out_labels.extend([int(cls)] * n_orig)

        if target_per_class is not None and n_orig < target_per_class:
            n_needed = target_per_class - n_orig
        else:
            n_needed = n_orig * (augmentation_factor - 1)

        if n_needed <= 0:
            continue

        for _ in range(n_needed):
            idx = rng.integers(0, n_orig)
            base = cls_feats[idx]
            aug = _augment_landmarks_sample(base, rng)
            out_feats.append(aug.reshape(1, -1))
            out_labels.append(int(cls))

    return (
        np.concatenate([f if f.ndim == 2 else f.reshape(1, -1) for f in out_feats]),
        np.array(out_labels, dtype=np.int64),
    )


def main():
    """Main preparation function"""
    print("=" * 60)
    print("TSL Dataset Preparation")
    print("=" * 60)

    features, labels, class_names = load_tsl_dataset()

    if features is None:
        print("\nFailed to load dataset.")
        print("Please ensure the dataset is downloaded:")
        print("  python scripts/download_data.py")
        return

    # Split ORIGINALS first, then augment ONLY the train fold. Augmenting the
    # whole dataset before splitting would scatter augmented siblings of the
    # same original across both folds (data leakage -> inflated val accuracy).
    from sklearn.model_selection import train_test_split
    print("\nSplitting originals 80/20 (stratified, random_state=42)...")
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
    except ValueError:
        # Some classes may have too few samples for stratification
        X_train, X_val, y_train, y_val = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
    print(f"Train fold: {len(X_train)} | Val fold: {len(X_val)}")

    print("\nBalancing + augmenting TRAIN fold only (target 200/class)...")
    aug_features, aug_labels = create_augmented_data(
        X_train, y_train, augmentation_factor=2, target_per_class=200
    )
    print(f"Original size: {len(features)}")
    print(f"Train fold after balancing+augmentation: {len(aug_features)}")
    from collections import Counter
    counts = Counter(aug_labels.tolist())
    print(f"Per-class range after balancing: min={min(counts.values())} max={max(counts.values())}")

    print("\nSaving processed data...")
    # The unaugmented files keep their original meaning: the raw originals.
    save_processed_data(features, labels, class_names)

    # Explicit splits: train = augmented originals, val = held-out originals.
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DATA_DIR / "X_train.npy", aug_features)
    np.save(PROCESSED_DATA_DIR / "y_train.npy", aug_labels)
    np.save(PROCESSED_DATA_DIR / "X_val.npy", X_val)
    np.save(PROCESSED_DATA_DIR / "y_val.npy", y_val)

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print(f"\nProcessed data location: {PROCESSED_DATA_DIR}")
    print("\nNext step:")
    print("  python -m src.models.train")


if __name__ == "__main__":
    main()
