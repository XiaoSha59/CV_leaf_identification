import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import (
    DATA_DIR,
    METRICS_DIR,
    RANDOM_STATE,
    TRAIN_SPLIT_PATH,
    VAL_SPLIT_PATH,
)
from src.data_loader import load_split, verify_image_paths
from src.features import extract_hog, extract_lbp_histogram
from src.preprocess import preprocess_image


def extract_raw_representations(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract raw HOG and LBP features for a dataframe split."""
    hog_vectors = []
    lbp_vectors = []

    for _, row in df.iterrows():
        _, gray_norm = preprocess_image(row["image_path"])
        hog_vectors.append(extract_hog(gray_norm))
        lbp_vectors.append(extract_lbp_histogram(gray_norm))

    X_hog = np.vstack(hog_vectors)
    X_lbp = np.vstack(lbp_vectors)
    y = df["label"].to_numpy(dtype=np.int64)

    return X_hog, X_lbp, y


def sweep_pca_k(
    X_hog_train: np.ndarray,
    X_lbp_train: np.ndarray,
    y_train: np.ndarray,
    X_hog_val: np.ndarray,
    X_lbp_val: np.ndarray,
    y_val: np.ndarray,
    candidate_k: list[int] | None = None,
) -> pd.DataFrame:
    """Evaluate candidate k values on the validation set."""
    if candidate_k is None:
        candidate_k = [16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 256, 384, 512]

    # Fit full PCA to compute cumulative variance
    print("\nFitting full PCA on training HOG features to compute variance distribution...")
    full_pca = PCA(random_state=RANDOM_STATE).fit(X_hog_train)
    cum_variance = np.cumsum(full_pca.explained_variance_ratio_)

    results = []

    print("\n" + "=" * 80)
    print(f"{'k (PCA)':<10} {'Total Dim':<12} {'Variance %':<14} {'Val Acc %':<12} {'Val Macro F1':<14} {'Time (s)':<10}")
    print("=" * 80)

    for k in candidate_k:
        if k > X_hog_train.shape[0] or k > X_hog_train.shape[1]:
            continue

        start_t = perf_counter()

        # 1. Fit StandardScaler + PCA on train HOG and project both train & val
        hog_pca_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=k, random_state=RANDOM_STATE)),
        ])
        X_hog_tr_pca = hog_pca_pipe.fit_transform(X_hog_train)
        X_hog_va_pca = hog_pca_pipe.transform(X_hog_val)

        # 2. Concatenate with LBP (59D)
        X_tr_fused = np.hstack([X_hog_tr_pca, X_lbp_train])
        X_va_fused = np.hstack([X_hog_va_pca, X_lbp_val])

        # 3. Fit standard RBF-SVM pipeline
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(C=10.0, kernel="rbf", gamma="scale", random_state=RANDOM_STATE)),
        ])
        model.fit(X_tr_fused, y_train)
        y_val_pred = model.predict(X_va_fused)

        elapsed = perf_counter() - start_t

        acc = accuracy_score(y_val, y_val_pred) * 100.0
        f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
        var_pct = float(cum_variance[k - 1] * 100.0)
        total_dim = k + X_lbp_train.shape[1]

        results.append({
            "k": k,
            "total_dim": total_dim,
            "variance_pct": round(var_pct, 2),
            "val_accuracy": round(acc, 2),
            "val_macro_f1": round(f1, 4),
            "elapsed_seconds": round(elapsed, 2),
        })

        print(f"{k:<10} {total_dim:<12} {var_pct:>10.2f}% {acc:>10.2f}% {f1:>12.4f} {elapsed:>9.2f}s")

    print("=" * 80)

    results_df = pd.DataFrame(results)
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Find optimal PCA dimension k for HOG features.")
    parser.add_argument("--save-report", action="store_true", help="Save results table as JSON report.")
    args = parser.parse_args()

    train_df = load_split(TRAIN_SPLIT_PATH)
    val_df = load_split(VAL_SPLIT_PATH)

    verify_image_paths(train_df)
    verify_image_paths(val_df)

    print(f"Extracting features from {len(train_df)} train samples and {len(val_df)} val samples...")
    X_hog_tr, X_lbp_tr, y_tr = extract_raw_representations(train_df)
    X_hog_va, X_lbp_va, y_va = extract_raw_representations(val_df)

    results_df = sweep_pca_k(
        X_hog_tr, X_lbp_tr, y_tr,
        X_hog_va, X_lbp_va, y_va,
    )

    # Optimal k selection
    best_row = results_df.sort_values(by=["val_accuracy", "val_macro_f1", "k"], ascending=[False, False, True]).iloc[0]
    print(f"\nOPTIMAL PCA DIMENSION: k = {int(best_row['k'])} (Total fused dimensions: {int(best_row['total_dim'])}D)")
    print(f"  Validation Accuracy: {best_row['val_accuracy']:.2f}%")
    print(f"  Validation Macro F1: {best_row['val_macro_f1']:.4f}")
    print(f"  Variance Retained:   {best_row['variance_pct']:.2f}%")

    if args.save_report:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = METRICS_DIR / "pca_k_sweep_results.json"
        results_df.to_json(report_path, orient="records", indent=2)
        print(f"Saved sweep report to: {report_path}")


if __name__ == "__main__":
    main()
