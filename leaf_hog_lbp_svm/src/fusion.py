import numpy as np
from sklearn.decomposition import PCA

from src.features import extract_hog, extract_lbp_histogram
from src.pca_reduction import transform_pca


def concatenate_features(
    feature_matrix_a: np.ndarray,
    feature_matrix_b: np.ndarray,
) -> np.ndarray:
    """Concatenate two feature matrices or vectors horizontally."""
    if feature_matrix_a.ndim == 1 and feature_matrix_b.ndim == 1:
        return np.concatenate([feature_matrix_a, feature_matrix_b]).astype(np.float32)
    return np.hstack([feature_matrix_a, feature_matrix_b]).astype(np.float32)


def fuse_hog_pca_lbp(
    X_hog: np.ndarray,
    X_lbp: np.ndarray,
    pca_model: PCA,
) -> np.ndarray:
    """Apply PCA on HOG features and concatenate with LBP features."""
    X_hog_reduced = transform_pca(X_hog, pca_model)
    return concatenate_features(X_hog_reduced, X_lbp)


def extract_fused_features_single(
    image_gray: np.ndarray,
    pca_model: PCA,
) -> np.ndarray:
    """Extract fused HOG-PCA and LBP feature vector for a single image."""
    hog_feat = extract_hog(image_gray).reshape(1, -1)
    lbp_feat = extract_lbp_histogram(image_gray).reshape(1, -1)

    hog_pca = transform_pca(hog_feat, pca_model)
    fused = np.concatenate([hog_pca.ravel(), lbp_feat.ravel()]).astype(np.float32)
    return fused
