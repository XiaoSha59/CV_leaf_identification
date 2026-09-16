import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CLASS_RANGES, FIGURES_DIR, IMAGE_SIZE, LABELS_PATH, RAW_DATA_DIR
from src.data_loader import load_split
from src.features import (
    extract_hog_with_visualization,
    extract_lbp_histogram,
    extract_lbp_map,
)
from src.preprocess import load_bgr, preprocess_image


def minmax_normalize(image: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 1] for display."""
    image_min = image.min()
    image_max = image.max()
    if image_max - image_min < 1e-12:
        return np.zeros_like(image)
    return (image - image_min) / (image_max - image_min)


def render_dataset_grid(output_path: Path | None = None) -> Path:
    """Render a showcase grid of leaf samples from dataset classes."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = FIGURES_DIR / "dataset_grid.png"

    if LABELS_PATH.exists():
        labels_df = pd.read_csv(LABELS_PATH)
        samples = labels_df.groupby("class_name").first().reset_index()
    else:
        samples = pd.DataFrame(
            [{"class_name": name, "image_path": str(RAW_DATA_DIR / f"{start}.jpg")}
             for _, name, start, _ in CLASS_RANGES]
        )

    num_classes = len(samples)
    cols = 8 if num_classes > 15 else 5
    rows = int(np.ceil(num_classes / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5))
    axes = np.atleast_1d(axes).flatten()

    for idx, row in samples.iterrows():
        img_p = Path(row["image_path"])
        ax = axes[idx]
        if img_p.exists():
            bgr = load_bgr(img_p)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
        ax.set_title(
            row["class_name"].replace("_", " ").title()[:14],
            fontsize=8,
            fontweight="bold",
        )
        ax.axis("off")

    for u in range(num_classes, len(axes)):
        axes[u].axis("off")

    fig.suptitle(
        f"Flavia Botanical Species Gallery ({num_classes} Classes)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dataset grid: {output_path}")
    return output_path


def render_split_overview(output_path: Path | None = None) -> Path:
    """Render a visual summary chart of dataset train/val/test splits."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = FIGURES_DIR / "split_overview.png"

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    total = len(train_df) + len(val_df) + len(test_df)
    counts = [len(train_df), len(val_df), len(test_df)]
    labels = [
        f"Train ({len(train_df)} - {len(train_df)/total*100:.1f}%)",
        f"Validation ({len(val_df)} - {len(val_df)/total*100:.1f}%)",
        f"Test ({len(test_df)} - {len(test_df)/total*100:.1f}%)",
    ]
    colors = ["#4C78A8", "#59A14F", "#F28E2B"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].pie(
        counts,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor="w"),
    )
    axes[0].set_title("Stratified Split Proportions (70/15/15)", fontsize=11, fontweight="bold")

    axes[1].bar(["Train", "Validation", "Test"], counts, color=colors, width=0.5)
    for i, c in enumerate(counts):
        axes[1].text(i, c + 15, f"{c} samples", ha="center", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Number of Samples", fontsize=10)
    axes[1].set_ylim(0, max(counts) * 1.15)
    axes[1].set_title("Sample Distribution per Partition", fontsize=11, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"Dataset Split Overview (Total: {total} Samples across {train_df['label'].nunique()} Classes)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved split overview: {output_path}")
    return output_path


def render_feature_visualization(image_path: Path, output_path: Path | None = None) -> Path:
    """Create a 4-panel (A, B, C, D) HOG and Uniform LBP visualization."""
    image_bgr, image_gray = preprocess_image(image_path)

    hog_vector, hog_image = extract_hog_with_visualization(image_gray)
    lbp_map = extract_lbp_map(image_gray)
    lbp_histogram = extract_lbp_histogram(image_gray)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = FIGURES_DIR / "hog_lbp_feature_visualization.png"

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    # Panel A: Normalized leaf image
    axes[0].imshow(image_gray, cmap="gray")
    axes[0].set_title(
        f"(A) Normalized leaf image\nTarget: {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]} px",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].axis("off")

    # Panel B: HOG visualization
    axes[1].imshow(minmax_normalize(hog_image), cmap="gray")
    axes[1].set_title(
        f"(B) HOG visualization\n({len(hog_vector)} shape/edge features)",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].axis("off")

    # Panel C: Uniform LBP map
    axes[2].imshow(lbp_map, cmap="viridis")
    axes[2].set_title(
        "(C) Uniform LBP map\n(Local texture patterns)",
        fontsize=11,
        fontweight="bold",
    )
    axes[2].axis("off")

    # Panel D: Uniform LBP histogram (10 bins)
    axes[3].bar(
        np.arange(len(lbp_histogram)),
        lbp_histogram,
        color="#2b5c8f",
        width=0.65,
    )
    axes[3].set_title(
        f"(D) Uniform LBP histogram\n({len(lbp_histogram)} bins texture summary)",
        fontsize=11,
        fontweight="bold",
    )
    axes[3].set_xlabel("LBP Bin Code", fontsize=10)
    axes[3].set_ylabel("Normalized Frequency", fontsize=10)
    axes[3].grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Handcrafted Feature Representation: {image_path.name} "
        f"(HOG: {len(hog_vector)}-D + LBP: {len(lbp_histogram)}-D = {len(hog_vector) + len(lbp_histogram)}-D Hybrid)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved HOG & LBP feature visualization: {output_path}")
    print(f"HOG dimension: {len(hog_vector)}")
    print(f"LBP dimension: {len(lbp_histogram)}")
    print(f"Concatenated hybrid dimension: {len(hog_vector) + len(lbp_histogram)}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive visualization tool for Flavia leaf identification."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="feature-demo",
        choices=["feature-demo", "dataset-grid", "split-overview"],
        help="Visualization mode to run.",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=Path("data/raw/Flavia/1001.jpg"),
        help="Example image path for feature visualization.",
    )
    args = parser.parse_args()

    if args.mode == "dataset-grid":
        render_dataset_grid()
    elif args.mode == "split-overview":
        render_split_overview()
    else:
        if not args.image.exists():
            raise FileNotFoundError(f"Image not found: {args.image}")
        render_feature_visualization(args.image)


if __name__ == "__main__":
    main()