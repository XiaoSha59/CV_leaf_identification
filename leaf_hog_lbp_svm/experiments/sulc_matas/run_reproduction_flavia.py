import sys
from pathlib import Path
from time import perf_counter
import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.sulc_matas.classifier import FfirstRegionClassifier, fuse_probabilities
from experiments.sulc_matas.multiscale import extract_all_regions_multiscale
from src.config import RANDOM_STATE, TEST_SPLIT_PATH, TRAIN_SPLIT_PATH, VAL_SPLIT_PATH


def _process_one_image(img_rel_path: str):
    img_path = PROJECT_ROOT / img_rel_path
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    return extract_all_regions_multiscale(
        image_gray=img_gray,
        n_conc=3,
        c=6,
        P=8,
        target_width=300,
    )


def extract_dataset_descriptors_parallel(
    df: pd.DataFrame,
    split_name: str,
    n_jobs: int = -1,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]], list[list[np.ndarray]]]:
    total = len(df)
    print(f"\nExtracting Multi-Scale Descriptors for [{split_name.upper()}] ({total} images on all CPU cores)...", flush=True)
    t0 = perf_counter()

    paths = df["image_path"].tolist()
    results = Parallel(n_jobs=n_jobs, batch_size=16)(
        delayed(_process_one_image)(p) for p in paths
    )

    elapsed = perf_counter() - t0
    speed = total / max(0.1, elapsed)
    print(f"  [{split_name}] Completed {total} images in {elapsed:.1f}s ({speed:.1f} img/s)!", flush=True)

    all_list = [r["all"] for r in results]
    interior_list = [r["interior"] for r in results]
    border_list = [r["border"] for r in results]

    return all_list, interior_list, border_list


def run_flavia_reproduction(C_param: float = 1000.0):
    print("=" * 85, flush=True)
    print("REPRODUCTION: Milan Sulc & Jiri Matas (ECCV 2014) - Texture-Based Leaf Identification", flush=True)
    print("=" * 85, flush=True)
    
    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    val_df = pd.read_csv(VAL_SPLIT_PATH)
    test_df = pd.read_csv(TEST_SPLIT_PATH)
    
    train_val_df = pd.concat([train_df, val_df], ignore_index=True).sample(
        frac=1.0, random_state=RANDOM_STATE
    ).reset_index(drop=True)
    
    y_train = train_val_df["label"].to_numpy(dtype=np.int64)
    y_test = test_df["label"].to_numpy(dtype=np.int64)
    
    print(f"Dataset split: Train+Val = {len(train_val_df)} images | Test = {len(test_df)} images (32 classes)", flush=True)
    
    # 1. Extract Descriptors in Parallel
    X_tr_all, X_tr_interior, X_tr_border = extract_dataset_descriptors_parallel(train_val_df, "Train+Val", n_jobs=-1)
    X_te_all, X_te_interior, X_te_border = extract_dataset_descriptors_parallel(test_df, "Test", n_jobs=-1)
    
    # 2. Train Classifiers
    print("\n--- Training Ffirst Chi-Square Classifiers ---", flush=True)
    print("Fitting Ffirst-All model...", flush=True)
    clf_all = FfirstRegionClassifier(C=C_param).fit(X_tr_all, y_train)
    
    print("Fitting Ffirst-Interior model...", flush=True)
    clf_interior = FfirstRegionClassifier(C=C_param).fit(X_tr_interior, y_train)
    
    print("Fitting Ffirst-Border model...", flush=True)
    clf_border = FfirstRegionClassifier(C=C_param).fit(X_tr_border, y_train)
    
    # 3. Predict Posterior Probabilities on Test Set
    print("\n--- Evaluating on Held-out Test Set (287 images) ---", flush=True)
    probs_all = clf_all.predict_proba_dataset(X_te_all)
    probs_interior = clf_interior.predict_proba_dataset(X_te_interior)
    probs_border = clf_border.predict_proba_dataset(X_te_border)
    
    preds_all = np.argmax(probs_all, axis=1)
    preds_interior = np.argmax(probs_interior, axis=1)
    preds_border = np.argmax(probs_border, axis=1)
    
    # Late Fusion: Sum and Product
    _, preds_fused_sum = fuse_probabilities(probs_interior, probs_border, fusion_method="sum")
    _, preds_fused_prod = fuse_probabilities(probs_interior, probs_border, fusion_method="product")
    
    # 4. Compute Benchmark Metrics
    variants = [
        ("1. Ffirst-a (All Leaf Pixels)", preds_all),
        ("2. Ffirst-i (Leaf Interior Only)", preds_interior),
        ("3. Ffirst-b (Leaf Border Only)", preds_border),
        ("4. Ffirst-ib-sum (Sum Fusion: Interior + Border)", preds_fused_sum),
        ("5. Ffirst-ib-prod (Product Fusion: Interior x Border)", preds_fused_prod),
    ]
    
    results = []
    for name, preds in variants:
        acc = accuracy_score(y_test, preds) * 100.0
        f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        correct = np.sum(y_test == preds)
        total = len(y_test)
        
        results.append({
            "Method Variant": name,
            "Accuracy": f"{acc:.2f}%",
            "Macro F1": f"{f1:.4f}",
            "Correct": f"{correct}/{total}",
            "Errors": f"{total - correct}",
        })
        
    print("\n" + "=" * 85, flush=True)
    print("SULC & MATAS (ECCV 2014) REPRODUCTION RESULTS ON FLAVIA TEST SET (287 IMAGES)", flush=True)
    print("=" * 85, flush=True)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False), flush=True)
    print("=" * 85, flush=True)
    
    return results_df


if __name__ == "__main__":
    run_flavia_reproduction(C_param=1000.0)
