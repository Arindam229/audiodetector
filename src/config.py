"""Central configuration shared across preprocessing, training, and inference.

All paths are resolved relative to the project root (the parent of `src/`)
so that scripts work regardless of the current working directory.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "for-norm" / "for-norm"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
APP_DIR = PROJECT_ROOT / "app"

CHECKPOINT_PATH = MODELS_DIR / "detector.pt"
METRICS_REPORT_PATH = MODELS_DIR / "metrics_report.json"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = MODELS_DIR / "roc_curve.png"

# ---------------------------------------------------------------------------
# Dataset layout / class mapping
# ---------------------------------------------------------------------------
# Split directory name -> processed file stem
SPLITS = {
    "training": "train",
    "validation": "val",
    "testing": "test",
}

# Class subfolder name (lowercased) -> integer label
# 1 = Deepfake / AI-Generated, 0 = Genuine / Human speech
CLASS_MAP = {
    "fake": 1,
    "spoof": 1,
    "real": 0,
    "genuine": 0,
    "bonafide": 0,
}

CLASS_NAMES = {0: "GENUINE (HUMAN)", 1: "DEEPFAKE (AI-GENERATED)"}

# Maximum number of files to process per class for each split. This keeps the
# end-to-end pipeline fast and reproducible while still drawing on tens of
# thousands of recordings. Set a value to `None` to use every file.
MAX_FILES_PER_CLASS = {
    "training": None,
    "validation": None,
    "testing": None,
}

# ---------------------------------------------------------------------------
# Audio / feature extraction parameters
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
DURATION_SECONDS = 4.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION_SECONDS)  # 64000

N_LFCC = 20          # base LFCC coefficients
N_FILTER = 40        # number of linear filterbank channels
N_FFT = 512
WIN_LENGTH = 400     # 25 ms @ 16 kHz
HOP_LENGTH = 160     # 10 ms @ 16 kHz

# Derived shapes (kept here so every module agrees on tensor sizes)
NUM_FRAMES = NUM_SAMPLES // HOP_LENGTH + 1   # 401 frames (center=True STFT)
FEATURE_DIM = N_LFCC * 3                     # LFCC + delta + delta-delta = 60

# ---------------------------------------------------------------------------
# Model / training hyperparameters (defaults; train.py may search around these)
# ---------------------------------------------------------------------------
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
CNN_CHANNELS = (64, 128)
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
BIDIRECTIONAL = True

# Validation thresholds the trained model must satisfy
TARGET_ACCURACY = 0.80
TARGET_EER = 0.12
TARGET_F1 = 0.80
TARGET_PER_CLASS_ACCURACY = 0.75

RANDOM_SEED = 42
