"""Shared paths for the alternative_models experiments."""
from pathlib import Path

ALT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ALT_ROOT.parent

DATA_DIR = ALT_ROOT / "data"
MODELS_DIR = ALT_ROOT / "models"
RESULTS_DIR = ALT_ROOT / "results"
PLOTS_DIR = ALT_ROOT / "plots"
TRAIN_CURVES_DIR = PLOTS_DIR / "training_curves"
CONFUSION_DIR = PLOTS_DIR / "confusion_matrices"
COMPARISONS_DIR = PLOTS_DIR / "comparisons"
ROBUSTNESS_DIR = PLOTS_DIR / "robustness"

DATASET_ROOT = REPO_ROOT / "data" / "raw" / "tsl_dataset" / "data" / "Data"

RAW_LANDMARKS_PATH = DATA_DIR / "raw_landmarks.npz"
DATASET_PATH = DATA_DIR / "dataset.npz"

GESTURE_EMBEDDER_PATH = (
    Path(__file__).parent.parent / "assets" / "gesture_embedder.tflite"
)

SEED = 42
NUM_CLASSES = 57

for d in [
    DATA_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, TRAIN_CURVES_DIR,
    CONFUSION_DIR, COMPARISONS_DIR, ROBUSTNESS_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)