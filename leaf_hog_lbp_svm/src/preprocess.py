import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.config import FIGURES_DIR, IMAGE_SIZE


def load_bgr(image_path: str | Path) -> np.ndarray:
    """Load an image as a BGR uint8 array."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return image_bgr


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB for matplotlib visualization."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def create_leaf_mask(image_gray: np.ndarray) -> np.ndarray:
    """Segment leaf foreground from white background using Otsu thresholding."""
    _, thresh = cv2.threshold(
        image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return thresh


def get_largest_contour(mask: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Extract the largest external contour representing the primary leaf blade."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, 0.0
    leaf_cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(leaf_cnt))
    return leaf_cnt, area


def align_principal_axis(
    image_gray: np.ndarray,
    mask: np.ndarray,
    leaf_cnt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Compute central moments and perform affine rotation to align major axis vertically."""
    moments = cv2.moments(leaf_cnt)
    if moments["m00"] == 0:
        return image_gray, mask, 0.0, 0.0, 0.0

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    mu20 = moments["mu20"] / moments["m00"]
    mu02 = moments["mu02"] / moments["m00"]
    mu11 = moments["mu11"] / moments["m00"]

    # Principal axis orientation angle
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    angle_deg = float(np.degrees(theta))

    # Align major axis vertically
    rot_deg = angle_deg - 90.0 if abs(angle_deg) > 45 else angle_deg

    h, w = image_gray.shape
    rot_mat = cv2.getRotationMatrix2D((cx, cy), rot_deg, 1.0)
    # Translate centroid to image center
    rot_mat[0, 2] += w / 2.0 - cx
    rot_mat[1, 2] += h / 2.0 - cy

    aligned_gray = cv2.warpAffine(image_gray, rot_mat, (w, h), borderValue=255)
    aligned_mask = cv2.warpAffine(mask, rot_mat, (w, h), borderValue=0)

    return aligned_gray, aligned_mask, rot_deg, cx, cy


def center_leaf(aligned_gray: np.ndarray, aligned_mask: np.ndarray) -> np.ndarray:
    """Crop bounding box around aligned leaf contour."""
    contours, _ = cv2.findContours(
        aligned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        aligned_cnt = max(contours, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(aligned_cnt)
        if bw > 10 and bh > 10:
            return aligned_gray[by : by + bh, bx : bx + bw]
    return aligned_gray


def normalize_leaf(
    cropped_gray: np.ndarray,
    target_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Standardize final leaf frame to fixed dimensions (100x134)."""
    return cv2.resize(cropped_gray, target_size, interpolation=cv2.INTER_AREA)


def segment_and_normalize_leaf(
    image_bgr: np.ndarray,
    target_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """End-to-end morphological preprocessing pipeline (Islam et al., 2019)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mask = create_leaf_mask(gray)
    leaf_cnt, area = get_largest_contour(mask)

    if leaf_cnt is None or area < 100:
        return cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)

    aligned_gray, aligned_mask, _, _, _ = align_principal_axis(
        gray, mask, leaf_cnt
    )
    cropped = center_leaf(aligned_gray, aligned_mask)
    return normalize_leaf(cropped, target_size=target_size)


def to_gray_resize(
    image_bgr: np.ndarray,
    target_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Convenience entry point for preprocessing."""
    return segment_and_normalize_leaf(image_bgr, target_size=target_size)


def preprocess_image(
    image_path: str | Path,
    target_size: tuple[int, int] = IMAGE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return original BGR and normalized grayscale image."""
    image_bgr = load_bgr(image_path)
    image_gray = segment_and_normalize_leaf(image_bgr, target_size=target_size)
    return image_bgr, image_gray


def save_preprocessing_steps_figure(
    image_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Save multi-panel step-by-step preprocessing visualization figure."""
    image_bgr = load_bgr(image_path)
    image_rgb = bgr_to_rgb(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    mask = create_leaf_mask(gray)
    leaf_cnt, area = get_largest_contour(mask)

    contour_rgb = image_rgb.copy()
    if leaf_cnt is not None:
        cv2.drawContours(contour_rgb, [leaf_cnt], -1, (0, 255, 0), 4)

    aligned_gray, aligned_mask, rot_deg, cx, cy = align_principal_axis(
        gray, mask, leaf_cnt if leaf_cnt is not None else np.array([])
    )
    cropped_gray = center_leaf(aligned_gray, aligned_mask)
    normalized = normalize_leaf(cropped_gray, target_size=IMAGE_SIZE)

    fig, axes = plt.subplots(1, 6, figsize=(22, 4.5))

    axes[0].imshow(image_rgb)
    axes[0].set_title("1. Original RGB", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("2. Otsu Mask", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(contour_rgb)
    axes[2].set_title("3. Largest Contour", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(aligned_gray, cmap="gray")
    axes[3].set_title(
        f"4. Aligned Leaf\n(Rot: {rot_deg:.1f}°)",
        fontsize=11,
        fontweight="bold",
    )
    axes[3].axis("off")

    axes[4].imshow(cropped_gray, cmap="gray")
    axes[4].set_title("5. Centered & Cropped", fontsize=11, fontweight="bold")
    axes[4].axis("off")

    axes[5].imshow(normalized, cmap="gray")
    axes[5].set_title(
        f"6. Normalized\nTarget: {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]} px",
        fontsize=11,
        fontweight="bold",
    )
    axes[5].axis("off")

    fig.suptitle(
        f"Morphological Preprocessing Pipeline: {Path(image_path).name}",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        out_file = FIGURES_DIR / "preprocessing_steps.png"
    else:
        out_file = Path(output_path)

    fig.savefig(str(out_file), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved preprocessing steps figure: {out_file}")
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run and visualize leaf morphological preprocessing steps."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("data/raw/Flavia/1001.jpg"),
        help="Input leaf image path.",
    )
    parser.add_argument(
        "--save-steps",
        action="store_true",
        help="Save multi-panel visualization of preprocessing steps to results/figures/preprocessing_steps.png",
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    print(f"Preprocessing image: {args.image}")
    _, gray_norm = preprocess_image(args.image)
    print(f"Normalized shape: {gray_norm.shape} (H x W)")

    if args.save_steps or True:
        save_preprocessing_steps_figure(args.image)


if __name__ == "__main__":
    main()
