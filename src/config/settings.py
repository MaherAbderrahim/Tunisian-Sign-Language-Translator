"""
Configuration settings for TSL Translator
"""

import os
from pathlib import Path

# Project paths
# settings.py lives at <repo>/src/config/settings.py, so PROJECT_ROOT is
# three parents up (config -> src -> repo).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"

# W7CH dataset layout after `scripts/download_data.py`:
#   data/raw/tsl_dataset/data/Data/<Category>/<class>/<image>.jpg
# NOTE: older code assumed a flat `data/raw/tsl_dataset/data/<class>/` layout;
# the real layout nests a `Data/` parent directory then 5 category sub-dirs.
DATASET_ROOT = RAW_DATA_DIR / "tsl_dataset" / "data" / "Data"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset URLs
TSL_DATASET_URL = "https://github.com/W7CH/First-ever-Tunisian-Sign-Language-Dataset"
ARASL_DATASET_URL = "https://huggingface.co/datasets/pain/ArASL_Database_Grayscale"

# Model settings
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
# Class count is derived from the dictionary (single source of truth) but kept
# here for backward compatibility with code that imports NUM_CLASSES directly.
# It MUST stay in sync with len(TSLDictionary()).
NUM_CLASSES = 57
LEARNING_RATE = 0.001
NUM_EPOCHS = 150
EARLY_STOP_PATIENCE = 20

# Inference settings
# Note: the MLP-over-landmarks model reaches ~99% val accuracy but per-frame
# confidence on real test images rarely exceeds 0.5 because a single still
# image is ambiguous. 0.3 keeps useful predictions while rejecting obvious
# noise;temporal smoothing on a video stream compensates for low single-frame
# confidence.
CONFIDENCE_THRESHOLD = 0.3
TEMPORAL_SMOOTHING_WINDOW = 5  # frames used for majority-vote smoothing
HISTORY_MAX_LEN = 100  # bounded translation history

# Continuous recognition (StreamingRecognizer)
STREAMING_STABLE_FRAMES = 4   # consecutive stable frames to emit a finalized sign
STREAMING_GAP_FRAMES = 6      # consecutive empty frames to flush a sentence
STREAMING_BUFFER_SIZE = 12    # number of recent predictions kept

# Output and translation languages
DEFAULT_OUTPUT_LANG = "en"
DEFAULT_TRANSLATE_TO_LANGS = ["en", "fr", "ar"]

# Upload limits (API)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

# MediaPipe settings
MEDIAPIPE_MODEL_COMPLEXITY = 1
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5
MEDIAPIPE_HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# Default trained checkpoint path
DEFAULT_MODEL_PATH = MODELS_DIR / "best_model.pth"
