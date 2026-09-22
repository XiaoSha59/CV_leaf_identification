from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_fill_holes


def segment_leaf_mask(image_gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(
        image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    filled = binary_fill_holes(binary > 0).astype(np.uint8) * 255
    return filled


def get_circular_kernel(radius: float) -> np.ndarray:
    r_int = max(1, int(np.ceil(radius)))
    y, x = np.ogrid[-r_int:r_int + 1, -r_int:r_int + 1]
    mask = (x * x + y * y) <= (radius * radius)
    return mask.astype(np.uint8)


def split_interior_border(
    leaf_mask: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    kernel = get_circular_kernel(radius)
    interior_mask = cv2.erode(leaf_mask, kernel, iterations=1)
    border_mask = cv2.bitwise_and(leaf_mask, cv2.bitwise_not(interior_mask))
    return interior_mask, border_mask


def visualize_leaf_regions(
    image_path: str | Path,
    output_path: str | Path = "results/figures/sulc_matas_segmentation_demo.png",
    radii: list[float] | None = None,
) -> Path:
    if radii is None:
        radii = [1.0, 2.8, 5.6, 11.3]

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    leaf_mask = segment_leaf_mask(img_gray)
    
    n_radii = len(radii)
    fig, axes = plt.subplots(1, n_radii + 1, figsize=(4 * (n_radii + 1), 4.5))
    
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")
    
    for idx, r in enumerate(radii, start=1):
        interior, border = split_interior_border(leaf_mask, radius=r)
        overlay = np.ones_like(img_rgb) * 255
        overlay[interior > 0] = [30, 80, 200]
        overlay[border > 0] = [220, 40, 40]
        
        axes[idx].imshow(overlay)
        axes[idx].set_title(f"Scale R = {r:.1f}\nBlue: Interior | Red: Border", fontsize=10, fontweight="bold")
        axes[idx].axis("off")
        
    fig.suptitle(
        f"Sulc & Matas (2014) Region Splitting: {Path(image_path).name}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_file), dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_file


if __name__ == "__main__":
    test_img = Path("data/raw/Flavia/1001.jpg")
    if test_img.exists():
        visualize_leaf_regions(test_img)
