import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.config import FIGURES_DIR, LBP_PARAMS
from src.features import (
    extract_hog_with_visualization,
    extract_lbp_histogram,
    extract_lbp_map,
)
from src.preprocess import preprocess_image


def minmax_normalize(image: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 1] for display only."""
    image_min = image.min()
    image_max = image.max()

    if image_max - image_min < 1e-12:
        return np.zeros_like(image)

    return (image - image_min) / (image_max - image_min)


def visualize_features(image_path: Path) -> Path:
    """Create and save a five-panel HOG/LBP visualization for one image."""
    image_bgr, image_gray = preprocess_image(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    hog_vector, hog_image = extract_hog_with_visualization(image_gray)
    lbp_map = extract_lbp_map(image_gray)

    lbp_histogram = extract_lbp_histogram(image_gray)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Original RGB")
    axes[0].axis("off")

    axes[1].imshow(image_gray, cmap="gray")
    axes[1].set_title(f"Normalized ({image_gray.shape[1]}x{image_gray.shape[0]})")
    axes[1].axis("off")

    axes[2].imshow(minmax_normalize(hog_image), cmap="gray")
    axes[2].set_title(f"HOG visualization\n{len(hog_vector)} features")
    axes[2].axis("off")

    axes[3].imshow(lbp_map, cmap="viridis")
    axes[3].set_title("Uniform LBP map")
    axes[3].axis("off")

    axes[4].bar(
        np.arange(len(lbp_histogram)),
        lbp_histogram,
        color="#1f77b4",
    )
    axes[4].set_title(f"LBP histogram\n{len(lbp_histogram)} features")
    axes[4].set_xlabel("LBP code")
    axes[4].set_ylabel("Normalized frequency")

    fig.suptitle(
        f"HOG + LBP Feature Extraction: {image_path.name}",
        fontsize=14,
    )
    fig.tight_layout()

    output_path = FIGURES_DIR / f"features_{image_path.stem}.png"
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved visualization: {output_path}")
    print(f"HOG feature dimension: {len(hog_vector)}")
    print(f"LBP feature dimension: {len(lbp_histogram)}")
    print(f"Combined feature dimension: {len(hog_vector) + len(lbp_histogram)}")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize HOG and Uniform LBP for one leaf image."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Example: data/raw/Flavia/1001.jpg",
    )
    args = parser.parse_args()

    visualize_features(args.image)


if __name__ == "__main__":
    main()