from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "Flavia"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT/ "models"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

LABELS_PATH = DATA_DIR / "labels.csv"
TRAIN_SPLIT_PATH = DATA_DIR / "train.csv"
VAL_SPLIT_PATH = DATA_DIR / "val.csv"
TEST_SPLIT_PATH = DATA_DIR / "test.csv"

IMAGE_SIZE = (128, 128)

HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": 'L2-Hys',
}

LBP_PARAMS = {
    "radius": 1,
    "n_points": 8,
    "method": 'uniform',
}

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASS_RANGES = [
    (0, "pubescent_bamboo", 1001, 1059),
    (1, "chinese_horse_chestnut", 1060, 1122),
    (2, "anhui_barberry", 1552, 1616),
    (3, "chinese_redbud", 1123, 1194),
    (4, "true_indigo", 1195, 1267),
    (5, "japanese_maple", 1268, 1323),
    (6, "nanmu", 1324, 1385),
    (7, "castor_aralia", 1386, 1437),
    (8, "chinese_cinnamon", 1497, 1551),
    (9, "goldenrain_tree", 1438, 1496),
]