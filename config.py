"""Central configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
TRAIN_PATH = DATA_DIR / "train_normal.csv"
TEST_PATH = DATA_DIR / "test_stream.csv"

FEATURES = ["temperature", "vibration"]
RANDOM_STATE = 42

# Model selection
CANDIDATE_STATES = range(1, 7)
N_RESTARTS = 5            # EM is sensitive to initialisation, keep the best of several fits
VALIDATION_FRACTION = 0.25

# Sliding-window detection
WINDOW_SIZE = 40          # samples per scored window
STRIDE = 5                # samples between consecutive windows
THRESHOLD_PERCENTILE = 1.0  # flag windows scoring below the 1st percentile of normal windows
