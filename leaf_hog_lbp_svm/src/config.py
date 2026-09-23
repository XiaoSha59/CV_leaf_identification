from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "Flavia"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

LABELS_PATH = DATA_DIR / "labels.csv"
TRAIN_SPLIT_PATH = DATA_DIR / "train.csv"
VAL_SPLIT_PATH = DATA_DIR / "val.csv"
TEST_SPLIT_PATH = DATA_DIR / "test.csv"

IMAGE_SIZE = (100, 134)

# HOG parameters (100x134 canvas -> 165 blocks x 36 = 5940 dimensions)
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2",
}

# Uniform LBP parameters (59 dimensions)
LBP_PARAMS = {
    "radius": 1,
    "n_points": 8,
    "method": "nri_uniform",
}

# PCA components for HOG reduction
PCA_HOG_COMPONENTS = 64

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

CLASS_RANGES = [
    (0, "pubescent_bamboo", 1001, 1059),
    (1, "chinese_horse_chestnut", 1060, 1122),
    (2, "chinese_redbud", 1123, 1194),
    (3, "true_indigo", 1195, 1267),
    (4, "japanese_maple", 1268, 1323),
    (5, "nanmu", 1324, 1385),
    (6, "castor_aralia", 1386, 1437),
    (7, "goldenrain_tree", 1438, 1496),
    (8, "chinese_cinnamon", 1497, 1551),
    (9, "anhui_barberry", 1552, 1616),
    (10, "big_fruited_holly", 2001, 2050),
    (11, "japanese_cheesewood", 2051, 2113),
    (12, "wintersweet", 2114, 2165),
    (13, "camphortree", 2166, 2230),
    (14, "japan_arrowwood", 2231, 2290),
    (15, "sweet_osmanthus", 2291, 2346),
    (16, "deodar", 2347, 2423),
    (17, "ginkgo", 2424, 2485),
    (18, "crape_myrtle", 2486, 2546),
    (19, "oleander", 2547, 2612),
    (20, "yew_plum_pine", 2616, 2675),
    (21, "japanese_flowering_cherry", 3001, 3055),
    (22, "glossy_privet", 3056, 3110),
    (23, "chinese_toon", 3111, 3175),
    (24, "peach", 3176, 3229),
    (25, "ford_woodlotus", 3230, 3281),
    (26, "trident_maple", 3282, 3334),
    (27, "beales_barberry", 3335, 3389),
    (28, "southern_magnolia", 3390, 3446),
    (29, "canadian_poplar", 3447, 3510),
    (30, "chinese_tulip_tree", 3511, 3563),
    (31, "tangerine", 3566, 3621),
]