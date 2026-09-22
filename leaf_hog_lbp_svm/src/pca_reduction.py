"""PCA Dimensionality Reduction Module for HOG Features.

This module provides functions to train, apply, save, and load PCA models
specifically designed to compress high-dimensional HOG feature vectors (5,940D)
into a compact representation (e.g., 64D or 128D) while eliminating redundancy.
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA

from src.config import MODELS_DIR, PCA_HOG_COMPONENTS, RANDOM_STATE


def fit_pca(
    X_features: np.ndarray,
    n_components: int | float = PCA_HOG_COMPONENTS,
    random_state: int = RANDOM_STATE,
) -> PCA:
    """Fit a PCA model on the training feature matrix."""
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_features)
    var_retained = float(np.sum(pca.explained_variance_ratio_) * 100)
    print(
        f"Fitted PCA: {X_features.shape[1]}D -> {pca.n_components_}D "
        f"({var_retained:.2f}% cumulative variance retained)"
    )
    return pca


def transform_pca(
    X_features: np.ndarray,
    pca_model: PCA,
) -> np.ndarray:
    """Project high-dimensional features onto fitted principal components."""
    return pca_model.transform(X_features).astype(np.float32)


def fit_transform_pca(
    X_features: np.ndarray,
    n_components: int | float = PCA_HOG_COMPONENTS,
    random_state: int = RANDOM_STATE,
) -> tuple[PCA, np.ndarray]:
    """Fit PCA on feature matrix and return both the model and transformed features."""
    pca = fit_pca(X_features, n_components=n_components, random_state=random_state)
    X_reduced = transform_pca(X_features, pca)
    return pca, X_reduced


def save_pca(
    pca_model: PCA,
    output_path: str | Path | None = None,
) -> Path:
    """Persist the fitted PCA transformer to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = MODELS_DIR / f"hog_pca_{pca_model.n_components_}.joblib"
    else:
        output_path = Path(output_path)

    joblib.dump(pca_model, output_path)
    print(f"Saved PCA model to: {output_path}")
    return output_path


def load_pca(model_path: str | Path) -> PCA:
    """Load a previously fitted PCA model from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"PCA model file not found: {path}")
    return joblib.load(path)
