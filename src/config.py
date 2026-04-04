"""
config.py
Central configuration for paths, environment variables, and debug settings
"""

from pathlib import Path
import os

# -----------------------------
# Optional dotenv support
# -----------------------------
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


# -----------------------------
# Base Directory
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]


# -----------------------------
# Load .env file if available
# -----------------------------
if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env", override=True)


# -----------------------------
# Paths
# -----------------------------
DATA_PATH = BASE_DIR / os.getenv(
    "DATA_PATH",
    "data/raw/telecom_churn_20000.csv"
)

PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_PATH = PROCESSED_DATA_DIR / "test.csv"

MODEL_DIR = BASE_DIR / "models"
_model_path_env = Path(os.getenv("MODEL_PATH", "models/best_model.pkl"))
if _model_path_env.is_absolute():
    MODEL_PATH = _model_path_env
else:
    # Accept both `best_model.pkl` and `models/best_model.pkl` in .env.
    MODEL_PATH = BASE_DIR / _model_path_env

PREPROCESSOR_PATH = MODEL_DIR / "preprocessing_pipeline.pkl"

FIG_DIR = BASE_DIR / "reports" / "figures"
REPORT_DIR = BASE_DIR / "reports"


# -----------------------------
# Create directories if missing
# -----------------------------
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Debug Settings
# -----------------------------
DEBUG = os.getenv("DEBUG", "False") == "True"
DEBUG_SAMPLE_SIZE = int(os.getenv("DEBUG_SAMPLE_SIZE", 5000))


# -----------------------------
# Training Settings
# -----------------------------
RANDOM_STATE = 42

# SMOTE
SMOTE_SAMPLING = float(os.getenv("SMOTE_SAMPLING", 0.5))

# Hyperparameter tuning
N_ITER = int(os.getenv("N_ITER", 20))
CV_FOLDS = int(os.getenv("CV_FOLDS", 3))


# -----------------------------
# Logging (optional)
# -----------------------------
VERBOSE = os.getenv("VERBOSE", "True") == "True"


# -----------------------------
# Debug Print
# -----------------------------
if DEBUG:
    print(f"DEBUG MODE ENABLED ({DEBUG_SAMPLE_SIZE} samples)")
