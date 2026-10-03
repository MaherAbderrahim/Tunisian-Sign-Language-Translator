"""Dataset preparation for the alternative-model experiments.

Protocol (documented in the README):

1. Extract MediaPipe landmarks from every dataset image in a deterministic
   order (sorted categories/classes/files), keeping BOTH the image-normalized
   landmarks and the world landmarks + handedness (needed by the pretrained
   gesture embedder).
2. Build the normalized representation used by the current pipeline
   (wrist-relative, scaled by hand size) — identical to
   ``src/models/landmark_extractor.py::_normalize_features``.
3. Stratified 80/10/10 split on the ORIGINAL samples (before augmentation)
   so augmented copies of a training image can never leak into
   validation/test.
4. Augment ONLY the training split (rotation, scale, translation, jitter in
   image space, same ranges and same RNG seed as the historical pipeline) and
   balance it to 200 samples/class.

The same augmented samples feed every model, guaranteeing fair comparison.
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter

import cv2
import numpy as np

from .paths import REPO_ROOT, DATASET_ROOT, RAW_LANDMARKS_PATH, DATASET_PATH, SEED
from .hand_graph import FINGER_CHAINS

sys.path.insert(0, str(REPO_ROOT))

from src.config.settings import IMAGE_SIZE  # noqa: E402
from src.languages.dictionary import TSLDictionary  # noqa: E402

HANDS = 2
JOINTS = 21
COORDS = 3
FLAT = HANDS * JOINTS * COORDS  # 126

# Augmentation ranges (same as scripts/prepare_data.py)
ANGLE_MAX_DEG = 20.0
SCALE_RANGE = (0.85, 1.15)
SHIFT_MAX = 0.1
JITTER_STD = 0.02
TARGET_PER_CLASS = 200
AUG_FACTOR = 2


# --------------------------------------------------------------------------
# Raw landmark extraction
# --------------------------------------------------------------------------
def _iter_dataset_images(dataset_root: Path):
    """Yield (category, class_folder, canonical_label, label_idx, [paths]) in
    the same order as scripts/prepare_data.py but with sorted file lists."""
    dictionary = TSLDictionary()
    class_names = dictionary.get_all_labels()
    class_to_index = {name: i for i, name in enumerate(class_names)}

    category_dirs = sorted(d for d in dataset_root.iterdir() if d.is_dir())
    if len(category_dirs) == 1 and category_dirs[0].name.lower() == "data":
        dataset_root = category_dirs[0]
        category_dirs = sorted(d for d in dataset_root.iterdir() if d.is_dir())

    for category_dir in category_dirs:
        for class_dir in sorted(d for d in category_dir.iterdir() if d.is_dir()):
            canonical = dictionary.canonical_label(class_dir.name)
            if canonical is None:
                continue
            exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
            paths = []
            for ext in exts:
                paths.extend(class_dir.glob(ext))
            paths = sorted(paths, key=lambda p: str(p).lower())
            yield category_dir.name, class_dir.name, canonical, class_to_index[canonical], paths


def extract_raw_landmarks(dataset_root: Path = None, force: bool = False):
    """Extract image landmarks, world landmarks and handedness per image.

    Returns dict of arrays (saved to RAW_LANDMARKS_PATH):
        img_landmarks: (N, 2, 21, 3)  image-normalized (x, y in [0,1], z depth)
        world_landmarks: (N, 2, 21, 3) world coords in meters
        handedness: (N, 2) handedness score
        labels: (N,) class index
        n_hands: (N, 2) 1/0 mask of which of the two slots hold a real hand
    """
    dataset_root = Path(dataset_root or DATASET_ROOT)
    if RAW_LANDMARKS_PATH.exists() and not force:
        return np.load(RAW_LANDMARKS_PATH)

    import mediapipe as mp

    model_path = REPO_ROOT / "models" / "hand_landmarker.task"
    base = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
    )
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    img_lm, world_lm, handed, labels, n_hands = [], [], [], [], []
    per_class = Counter()
    total = 0
    for category, folder, canonical, label_idx, paths in _iter_dataset_images(dataset_root):
        for p in paths:
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            frame = cv2.resize(frame, IMAGE_SIZE)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect(mp_image)
            if not result.hand_landmarks:
                continue
            hands = []
            world = []
            hand_ness = []
            for i, hlm in enumerate(result.hand_landmarks[:2]):
                hands.append(np.array([[lm.x, lm.y, lm.z] for lm in hlm], dtype=np.float32))
                wlm = result.hand_world_landmarks[i]
                world.append(np.array([[lm.x, lm.y, lm.z] for lm in wlm], dtype=np.float32))
                hand_ness.append(result.handedness[i][0].score)
            # pad to 2 hands
            while len(hands) < 2:
                hands.append(np.zeros((21, 3), dtype=np.float32))
                world.append(np.zeros((21, 3), dtype=np.float32))
                hand_ness.append(0.0)
            img_lm.append(np.stack(hands[:2]))
            world_lm.append(np.stack(world[:2]))
            handed.append(np.array(hand_ness[:2], dtype=np.float32))
            labels.append(label_idx)
            n_hands.append(np.array([1.0 if h.sum() > 0 else 0.0 for h in hands[:2]], dtype=np.float32))
            per_class[label_idx] += 1
            total += 1
        print(f"  {folder} -> {canonical}: {per_class[label_idx]} total")

    landmarker.close()
    out = {
        "img_landmarks": np.stack(img_lm),
        "world_landmarks": np.stack(world_lm),
        "handedness": np.stack(handed),
        "labels": np.array(labels, dtype=np.int64),
        "n_hands": np.stack(n_hands),
    }
    np.savez(RAW_LANDMARKS_PATH, **out)
    print(f"Saved {total} samples to {RAW_LANDMARKS_PATH}")
    return out


# --------------------------------------------------------------------------
# Normalization (identical to the current pipeline)
# --------------------------------------------------------------------------
def wrist_normalize(raw: np.ndarray) -> np.ndarray:
    """raw: (..., 2, 21, 3) -> (..., 2, 21, 3) wrist-relative, hand-size scaled.

    Same math as MediaPipeLandmarkExtractor._normalize_features:
    subtract wrist, divide by distance wrist -> middle-finger tip.
    """
    raw = np.asarray(raw, dtype=np.float32)
    out = np.zeros_like(raw)
    for b in range(raw.shape[-3]):
        hand = raw[..., b, :, :]
        rel = hand - hand[..., 0:1, :]
        size = np.linalg.norm(rel[..., 12, :], axis=-1, keepdims=True)
        size = np.where(size > 0, size, 1.0)
        out[..., b, :, :] = rel / size[..., None, :]
    return out


def flatten(x: np.ndarray) -> np.ndarray:
    """(..., 2, 21, 3) -> (..., 126)"""
    return np.asarray(x, dtype=np.float32).reshape(x.shape[:-3] + (FLAT,))


# --------------------------------------------------------------------------
# Geometric features (landmark-derived only)
# --------------------------------------------------------------------------
def geometric_features(raw: np.ndarray) -> np.ndarray:
    """Compute per-hand geometric features from image-normalized landmarks.

    raw: (..., 2, 21, 3). Returns (..., 2, 156) per hand:
      - 60 bone unit vectors  (20 bones x 3)
      - 15 cos(joint angles)  (5 fingers x 3 angles)
      - 3  palm normal unit vector
      - 78 = 60 + 15 + 3
    """
    raw = np.asarray(raw, dtype=np.float32)
    shape = raw.shape[:-3]
    raw2 = raw.reshape(-1, 2, 21, 3)
    out = np.zeros((raw2.shape[0], 2, 78), dtype=np.float32)

    bones = [(0, 1), (1, 2), (2, 3), (3, 4),
             (0, 5), (5, 6), (6, 7), (7, 8),
             (0, 9), (9, 10), (10, 11), (11, 12),
             (0, 13), (13, 14), (14, 15), (15, 16),
             (0, 17), (17, 18), (18, 19), (19, 20)]

    for h in range(2):
        hand = raw2[:, h]  # (N, 21, 3)
        # bone vectors
        for k, (i, j) in enumerate(bones):
            v = hand[:, j] - hand[:, i]
            n = np.linalg.norm(v, axis=-1, keepdims=True)
            n = np.where(n > 1e-6, n, 1.0)
            out[:, h, k * 3:(k + 1) * 3] = v / n
        # joint angles: at each middle joint of a finger, angle between
        # the two adjacent bones (cos).
        angle_pairs = [(1, 2, 3), (2, 3, 4),      # thumb
                       (5, 6, 7), (6, 7, 8),      # index
                       (9, 10, 11), (10, 11, 12), # middle
                       (13, 14, 15), (14, 15, 16),# ring
                       (17, 18, 19), (18, 19, 20)]  # pinky
        for k, (a, b, c) in enumerate(angle_pairs):
            v1 = hand[:, b] - hand[:, a]
            v2 = hand[:, c] - hand[:, b]
            n1 = np.linalg.norm(v1, axis=-1, keepdims=True)
            n2 = np.linalg.norm(v2, axis=-1, keepdims=True)
            n1 = np.where(n1 > 1e-6, n1, 1.0)
            n2 = np.where(n2 > 1e-6, n2, 1.0)
            cosang = np.sum(v1 / n1 * v2 / n2, axis=-1)
            out[:, h, 60 + k] = np.clip(cosang, -1.0, 1.0)
        # palm normal: cross product of wrist->middle-base and wrist->index-base
        v1 = hand[:, 9] - hand[:, 0]
        v2 = hand[:, 5] - hand[:, 0]
        nrm = np.cross(v1, v2)
        n = np.linalg.norm(nrm, axis=-1, keepdims=True)
        n = np.where(n > 1e-6, n, 1.0)
        out[:, h, 75:78] = nrm / n

    return out.reshape(shape + (2, 78))


def joint_features(raw: np.ndarray) -> np.ndarray:
    """Per-joint features for token/graph models: (..., 2, 21, 9).

    Channels: [x, y, z] image-normalized coords + incoming bone unit vector
    (toward the parent joint) + outgoing bone unit vector (toward the child).
    Missing fingers are zero vectors. Derived only from landmarks.
    """
    raw = np.asarray(raw, dtype=np.float32)
    shape = raw.shape[:-3]
    r = raw.reshape(-1, 2, 21, 3)
    out = np.zeros((r.shape[0], 2, 21, 9), dtype=np.float32)
    out[..., :3] = r

    chains = [list(c) for c in FINGER_CHAINS.values()]  # 5 chains
    for h in range(2):
        for chain in chains:
            for k in range(1, len(chain) - 1):
                i, j, nxt = chain[k - 1], chain[k], chain[k + 1]
                v_in = r[:, h, j] - r[:, h, i]
                v_out = r[:, h, nxt] - r[:, h, j]
                for name, v, off in [("in", v_in, 3), ("out", v_out, 6)]:
                    n = np.linalg.norm(v, axis=-1, keepdims=True)
                    n = np.where(n > 1e-6, n, 1.0)
                    out[:, h, j, off:off + 3] = v / n
            # thumb base (1) has only outgoing; tip (4) has only incoming
            i0, j0 = chain[0], chain[1]
            v = r[:, h, j0] - r[:, h, i0]
            n = np.linalg.norm(v, axis=-1, keepdims=True)
            n = np.where(n > 1e-6, n, 1.0)
            out[:, h, j0, 6:9] = v / n
            i0, j0 = chain[-2], chain[-1]
            v = r[:, h, j0] - r[:, h, i0]
            n = np.linalg.norm(v, axis=-1, keepdims=True)
            n = np.where(n > 1e-6, n, 1.0)
            out[:, h, j0, 3:6] = v / n
    return out.reshape(shape + (2, 21, 9))


# --------------------------------------------------------------------------
# Augmentation (image space, deterministic RNG like the historical pipeline)
# --------------------------------------------------------------------------
def _augment_pair(raw_pair: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """raw_pair: (2, 21, 3) image-normalized landmarks -> augmented copy."""
    f = raw_pair.copy()
    angle = float(rng.uniform(-ANGLE_MAX_DEG, ANGLE_MAX_DEG) * np.pi / 180.0)
    cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    scale = float(rng.uniform(*SCALE_RANGE))
    shift_x = float(rng.uniform(-SHIFT_MAX, SHIFT_MAX))
    shift_y = float(rng.uniform(-SHIFT_MAX, SHIFT_MAX))
    for h in range(2):
        if f[h].sum() == 0:
            continue
        xy = f[h, :, :2]
        xy = (xy - xy[0]) @ rot.T * scale + xy[0]
        xy[:, 0] += shift_x
        xy[:, 1] += shift_y
        f[h, :, :2] = xy
    noise = rng.normal(0.0, JITTER_STD, f.shape).astype(np.float32)
    return f + noise


def _augment_world(world_pair: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply the same geometric transform as `_augment_pair` to world
    landmarks (meters), with a small depth jitter (0.001 m ~ 1 mm)."""
    f = world_pair.copy()
    angle = float(rng.uniform(-ANGLE_MAX_DEG, ANGLE_MAX_DEG) * np.pi / 180.0)
    cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    scale = float(rng.uniform(*SCALE_RANGE))
    shift_x = float(rng.uniform(-SHIFT_MAX, SHIFT_MAX))
    shift_y = float(rng.uniform(-SHIFT_MAX, SHIFT_MAX))
    for h in range(2):
        if f[h].sum() == 0:
            continue
        xy = f[h, :, :2]
        xy = (xy - xy[0]) @ rot.T * scale + xy[0]
        xy[:, 0] += shift_x
        xy[:, 1] += shift_y
        f[h, :, :2] = xy
    noise = rng.normal(0.0, 0.001, f.shape).astype(np.float32)
    return f + noise


def augment_train_only(
    img_lm: np.ndarray, world_lm: np.ndarray, handedness: np.ndarray,
    labels: np.ndarray, n_hands: np.ndarray, train_idx: np.ndarray,
    target_per_class: int = TARGET_PER_CLASS,
    aug_factor: int = AUG_FACTOR,
):
    """Augment+balance the training split only (no leakage into val/test).

    Returns the same arrays for train/val/test splits, where train has been
    augmented to `target_per_class` samples per class.
    """
    rng = np.random.default_rng(seed=SEED)
    out = {"img_landmarks": [], "world_landmarks": [], "handedness": [],
           "labels": [], "n_hands": [], "split": []}
    for split_name, idx in [("train", train_idx)]:
        pass

    all_idx = np.arange(len(labels))
    val_test = np.setdiff1d(all_idx, train_idx)

    for split, idx in [("train", train_idx), ("val_test", val_test)]:
        pass

    # -- train: augment per class
    train_img, train_world, train_hand, train_lab, train_nh = [], [], [], [], []
    unique = np.unique(labels[train_idx])
    for cls in unique:
        mask = labels[train_idx] == cls
        idxs = train_idx[mask]
        n_orig = len(idxs)
        train_img.append(img_lm[idxs])
        train_world.append(world_lm[idxs])
        train_hand.append(handedness[idxs])
        train_lab.append(labels[idxs])
        train_nh.append(n_hands[idxs])
        n_needed = max(target_per_class - n_orig, 0) if n_orig < target_per_class else n_orig * (aug_factor - 1)
        for _ in range(n_needed):
            b = rng.integers(0, n_orig)
            base = img_lm[idxs[b]]
            aug = _augment_pair(base, rng)
            w_aug = _augment_world(world_lm[idxs[b]], rng)
            train_img.append(aug[None])
            train_world.append(w_aug[None])
            train_hand.append(handedness[idxs[b]][None])
            train_lab.append(labels[idxs[b]][None])
            train_nh.append(n_hands[idxs[b]][None])

    train = dict(
        img_landmarks=np.concatenate(train_img),
        world_landmarks=np.concatenate(train_world),
        handedness=np.concatenate(train_hand),
        labels=np.concatenate(train_lab),
        n_hands=np.concatenate(train_nh),
    )
    return train, val_test


def prepare_dataset(force: bool = False):
    """Full pipeline: extract -> split 80/10/10 (stratified, seed 42) ->
    augment train only. Returns dict of numpy arrays, cached to DATASET_PATH."""
    if DATASET_PATH.exists() and not force:
        return np.load(DATASET_PATH, allow_pickle=True)

    raw = extract_raw_landmarks(force=force)
    labels = raw["labels"]

    from sklearn.model_selection import train_test_split

    X_trval, X_test, y_trval, y_test = train_test_split(
        np.arange(len(labels)), labels, test_size=0.10,
        random_state=SEED, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trval, y_trval, test_size=1 / 9, random_state=SEED, stratify=y_trval)

    train, _ = augment_train_only(
        raw["img_landmarks"], raw["world_landmarks"], raw["handedness"],
        labels, raw["n_hands"], X_train)

    def subset(idx, include_world=True):
        d = {
            "img_landmarks": raw["img_landmarks"][idx],
            "labels": labels[idx],
            "n_hands": raw["n_hands"][idx],
        }
        if include_world:
            d["world_landmarks"] = raw["world_landmarks"][idx]
            d["handedness"] = raw["handedness"][idx]
        return d

    out = {
        "train": train,
        "val": subset(X_val),
        "test": subset(X_test),
    }
    np.savez(DATASET_PATH, allow_pickle=True,
             train_img=train["img_landmarks"], train_world=train["world_landmarks"],
             train_hand=train["handedness"], train_labels=train["labels"],
             train_nh=train["n_hands"],
             val_img=raw["img_landmarks"][X_val],
             val_world=raw["world_landmarks"][X_val],
             val_hand=raw["handedness"][X_val],
             val_labels=labels[X_val], val_nh=raw["n_hands"][X_val],
             test_img=raw["img_landmarks"][X_test],
             test_world=raw["world_landmarks"][X_test],
             test_hand=raw["handedness"][X_test],
             test_labels=labels[X_test], test_nh=raw["n_hands"][X_test])
    return out


def load_dataset(force_prepare: bool = False) -> dict:
    """Load (preparing if needed) and return the dataset as a dict with keys
    train/val/test, each containing img_landmarks (N,2,21,3), world_landmarks
    (N,2,21,3), handedness (N,2), labels (N,), n_hands (N,2)."""
    if not DATASET_PATH.exists() or force_prepare:
        prepare_dataset(force=force_prepare)
    z = np.load(DATASET_PATH, allow_pickle=True)
    return {
        "train": {"img_landmarks": z["train_img"], "labels": z["train_labels"],
                  "n_hands": z["train_nh"],
                  "world_landmarks": z["train_world"],
                  "handedness": z["train_hand"]},
        "val": {"img_landmarks": z["val_img"], "labels": z["val_labels"],
                "n_hands": z["val_nh"],
                "world_landmarks": z["val_world"],
                "handedness": z["val_hand"]},
        "test": {"img_landmarks": z["test_img"], "labels": z["test_labels"],
                 "n_hands": z["test_nh"],
                 "world_landmarks": z["test_world"],
                 "handedness": z["test_hand"]},
    }


def normalized_inputs(dataset: dict, with_geo: bool = False) -> dict:
    """Convert raw landmarks to model inputs.

    with_geo=False -> (N, 126) wrist-normalized flattened features.
    with_geo=True  -> (N, 282) = 126 normalized coords + 156 geometric feats.
    """
    out = {}
    for split, d in dataset.items():
        norm = wrist_normalize(d["img_landmarks"])
        feats = flatten(norm)
        if with_geo:
            geo = geometric_features(d["img_landmarks"]).reshape(
                len(d["labels"]), -1)
            feats = np.concatenate([feats, geo], axis=1)
        out[split] = {"features": feats, "labels": d["labels"],
                      "n_hands": d["n_hands"]}
    return out