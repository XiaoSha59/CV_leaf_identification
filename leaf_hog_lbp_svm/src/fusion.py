"""Feature Fusion Module (HOG PCA + LBP Concatenation).

This module handles feature concatenation after PCA dimensionality reduction:
1. Compresses high-dimensional HOG (5,940D) into k components (e.g., 64D).
2. Preserves the full discriminative power of Uniform LBP (59D).
3. Horizontally concatenates [HOG_PCA, LBP] into a balanced fused vector (123D).
"""

import numpy as np
from sklearn.decomposition import PCA

from src.features import extract_hog, extract_lbp_histogram
from src.pca_reduction import transform_pca


def concatenate_features(
    feature_matrix_a: np.ndarray,
    feature_matrix_b: np.ndarray,
) -> np.ndarray:
    """Horizontally concatenate two feature matrices or vectors."""
    if feature_matrix_a.ndim == 1 and feature_matrix_b.ndim == 1:
        return np.concatenate([feature_matrix_a, feature_matrix_b]).astype(np.float32)
    return np.hstack([feature_matrix_a, feature_matrix_b]).astype(np.float32)


def fuse_hog_pca_lbp(
    X_hog: np.ndarray,
    X_lbp: np.ndarray,
    pca_model: PCA,
) -> np.ndarray:
    """Project HOG feature matrix via fitted PCA and concatenate with LBP matrix.
    
    Args:
        X_hog: Raw HOG feature matrix (N, 5940).
        X_lbp: Normalized LBP feature matrix (N, 59).
        pca_model: Fitted sklearn PCA model (e.g., n_components=64).
        
    Returns:
        X_fused: Fused feature matrix (N, k + 59).
    """
    X_hog_reduced = transform_pca(X_hog, pca_model)
    return concatenate_features(X_hog_reduced, X_lbp)


def extract_fused_features_single(
    image_gray: np.ndarray,
    pca_model: PCA,
) -> np.ndarray:
    """Extract fused [HOG_PCA (k-dim) + LBP (59-dim)] vector for a single leaf image.
    
    Args:
        image_gray: Preprocessed grayscale uint8 leaf image (100x134).
        pca_model: Fitted sklearn PCA model.
        
    Returns:
        1D numpy float32 feature vector of length (k + 59).
    """
    hog_feat = extract_hog(image_gray).reshape(1, -1)
    lbp_feat = extract_lbp_histogram(image_gray).reshape(1, -1)
    
    hog_pca = transform_pca(hog_feat, pca_model)
    fused = np.concatenate([hog_pca.ravel(), lbp_feat.ravel()]).astype(np.float32)
    return fused
