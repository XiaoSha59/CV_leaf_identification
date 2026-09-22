import numpy as np
from skimage.feature import hog, local_binary_pattern

from src.config import HOG_PARAMS, LBP_PARAMS

def extract_hog(image_gray: np.ndarray) -> np.ndarray:
    """Extract a HOG feature vector from a grayscale image"""
    features = hog(
        image_gray,
        **HOG_PARAMS,
        feature_vector = True,
    )

    return features.astype(np.float32)

def extract_hog_with_visualization(
    image_gray: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract HOG features and a visualization image."""
    features, hog_image = hog(
        image_gray,
        **HOG_PARAMS,
        feature_vector = True, 
        visualize = True,
    )

    return features.astype(np.float32), hog_image.astype(np.float32)

def extract_lbp_map(image_gray: np.ndarray) -> np.ndarray:
    """Compute a uniform LBP code map from a grayscale uint8 image."""
    if image_gray.dtype != np.uint8:
        raise TypeError(f"Expected uint8 grayscale image, got {image_gray.dtype}")

    return local_binary_pattern(
        image_gray,
        LBP_PARAMS["n_points"],
        LBP_PARAMS["radius"],
        LBP_PARAMS["method"],
    )


def extract_lbp_histogram(image_gray: np.ndarray) -> np.ndarray:
    """Convert the uniform LBP map into a normalized histogram feature (59 bins for nri_uniform)."""
    lbp_map = extract_lbp_map(image_gray)

    p = LBP_PARAMS["n_points"]
    if LBP_PARAMS["method"] == "nri_uniform":
        n_bins = p * (p - 1) + 3  # 59 bins for P=8
    else:
        n_bins = p + 2

    histogram, _ = np.histogram(
        lbp_map.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True,
    )
    return histogram.astype(np.float32)


def extract_features(gray: np.ndarray, feature_set: str) -> np.ndarray:
    """Extract HOG, LBP, or concatenated HOG+LBP features."""
    if feature_set == "hog":
        return extract_hog(gray)

    if feature_set == "lbp":
        return extract_lbp_histogram(gray)

    if feature_set == "hog_lbp":
        return np.concatenate(
            [extract_hog(gray), extract_lbp_histogram(gray)]
        ).astype(np.float32)

    raise ValueError("feature_set must be one of: hog, lbp, hog_lbp")