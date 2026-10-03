"""
Prepare ArSL-31-Pilot dataset for TSL finger spelling.
Parses the CSV, converts to 126-dim landmark format, creates train/val/test splits.
"""
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ALPHABET_CSV = PROJECT_ROOT / "data" / "alphabet" / "ArSL_dataset.csv"
ALPHABET_PROCESSED = PROJECT_ROOT / "data" / "alphabet" / "processed"
ALPHABET_PROCESSED.mkdir(parents=True, exist_ok=True)

# ArSL labels -> our TSL alphabet labels
ARSL_TO_TSL = {
    "Alef": "alef", "Ba2": "ba", "Ta2": "ta", "Tha2": "tha",
    "Jim": "jim", "7a2": "ha", "Kha2": "kha", "Dal": "dal",
    "Thal": "dhal", "Ra2": "ra", "Zayn": "zay", "Sin": "sin",
    "Chin": "shin", "SSad": "sad", "DDad": "dad", "TTa2": "ta_h",
    "TTha2": "za", "3ayn": "ayn", "Ghayn": "ghayn", "Fa2": "fa",
    "9af": "qaf", "Kaf": "kaf", "Lam": "lam", "Mim": "mim",
    "Noon": "nun", "Ha2": "ha2", "Waw": "waw", "Ya2": "ya",
    # Special tokens
    "Space": "space", "Delete": "backspace", "Finish": "finish",
}

TSL_LABELS = [
    "alef","ba","ta","tha","jim","ha","kha","dal","dhal","ra","zay",
    "sin","shin","sad","dad","ta_h","za","ayn","ghayn","fa","qaf",
    "kaf","lam","mim","nun","ha2","waw","ya",
    "space","backspace","finish",
]

def parse_arsl_csv(csv_path):
    """Parse the ArSL-31-Pilot CSV file."""
    rows = []
    labels = []
    with open(csv_path, 'r') as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(';')
            if len(parts) < 43:
                continue
            sign = parts[0]
            coords = [float(x) for x in parts[1:43]]  # x0,y0..x20,y20
            rows.append(coords)
            labels.append(sign)
    return np.array(rows, dtype=np.float32), labels

def convert_to_126dim(coords_42):
    """Convert 42-dim (x0,y0..x20,y20) to 126-dim (x0,y0,z0..x20,y20,z0 for 2 hands)."""
    n = len(coords_42)
    features = []
    for i in range(n):
        row = coords_42[i]
        # Reshape to (21, 2) - 21 landmarks, x,y
        landmarks = np.array(row).reshape(21, 2)
        # Add z=0 for each landmark -> (21, 3)
        z_col = np.zeros((21, 1), dtype=np.float32)
        landmarks_3d = np.hstack([landmarks, z_col])  # (21, 3)
        # Flatten to 63 for one hand
        hand1 = landmarks_3d.flatten()  # (63,)
        # Pad second hand with zeros
        hand2 = np.zeros(63, dtype=np.float32)
        # Concatenate -> 126
        features.append(np.concatenate([hand1, hand2]))
    return np.array(features, dtype=np.float32)

def main():
    print("=" * 60)
    print("Preparing ArSL-31-Pilot Dataset")
    print("=" * 60)

    # Parse CSV
    coords, labels = parse_arsl_csv(ALPHABET_CSV)
    print(f"Parsed {len(coords)} samples, {len(set(labels))} classes")

    # Convert to 126-dim
    features = convert_to_126dim(coords)
    print(f"Features shape: {features.shape}")

    # Map labels
    tsl_labels = []
    skipped = 0
    valid_features = []
    for i, label in enumerate(labels):
        if label in ARSL_TO_TSL:
            tsl_label = ARSL_TO_TSL[label]
            if tsl_label in TSL_LABELS:
                tsl_labels.append(tsl_label)
                valid_features.append(features[i])
            else:
                skipped += 1
        else:
            skipped += 1

    features = np.array(valid_features, dtype=np.float32)
    label_indices = np.array([TSL_LABELS.index(l) for l in tsl_labels])
    print(f"Valid samples: {len(features)}, Skipped: {skipped}")

    # Class distribution
    unique, counts = np.unique(label_indices, return_counts=True)
    print("Class distribution:")
    for idx, count in zip(unique, counts):
        print(f"  {TSL_LABELS[idx]}: {count}")

    # Split: 80/10/10 stratified
    X_train, X_temp, y_train, y_temp = train_test_split(
        features, label_indices, test_size=0.2, random_state=42, stratify=label_indices
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print(f"\nSplits: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Augment training data
    rng = np.random.default_rng(42)
    X_aug, y_aug = [], []
    for cls in unique:
        mask = y_train == cls
        cls_feats = X_train[mask]
        n_needed = max(0, 200 - len(cls_feats))
        for _ in range(n_needed):
            idx = rng.integers(0, len(cls_feats))
            noise = rng.normal(0, 0.02, cls_feats[idx].shape).astype(np.float32)
            X_aug.append(cls_feats[idx] + noise)
            y_aug.append(cls)

    if X_aug:
        X_train = np.vstack([X_train, np.array(X_aug)])
        y_train = np.concatenate([y_train, np.array(y_aug)])
        print(f"After augmentation: train={len(X_train)}")

    # Save
    np.save(ALPHABET_PROCESSED / "features.npy", features)
    np.save(ALPHABET_PROCESSED / "labels.npy", label_indices)
    np.save(ALPHABET_PROCESSED / "X_train.npy", X_train)
    np.save(ALPHABET_PROCESSED / "y_train.npy", y_train)
    np.save(ALPHABET_PROCESSED / "X_val.npy", X_val)
    np.save(ALPHABET_PROCESSED / "y_val.npy", y_val)
    np.save(ALPHABET_PROCESSED / "X_test.npy", X_test)
    np.save(ALPHABET_PROCESSED / "y_test.npy", y_test)
    with open(ALPHABET_PROCESSED / "class_names.txt", "w", encoding="utf-8") as f:
        for name in TSL_LABELS:
            f.write(name + "\n")

    print(f"\nSaved to {ALPHABET_PROCESSED}")
    print("Done!")

if __name__ == "__main__":
    main()
