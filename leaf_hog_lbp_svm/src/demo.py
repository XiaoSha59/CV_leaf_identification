import argparse
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from src.config import FIGURES_DIR, MODELS_DIR, PREDICTIONS_DIR
from src.data_loader import load_split
from src.features import extract_features
from src.preprocess import preprocess_image


def load_final_model(feature_set: str = "hog_lbp"):
    """Load the trained RBF-SVM pipeline for the requested feature set."""
    if feature_set == "uniform_lbp":
        feature_set = "lbp"

    model_path = MODELS_DIR / f"final_{feature_set}_svm.joblib"
    if not model_path.exists():
        model_path = MODELS_DIR / f"{feature_set}_svm.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run `python -m src.evaluate` or `python -m src.train` first."
        )

    return joblib.load(model_path)


def get_class_names() -> list[str]:
    """Return class names ordered by numeric label from saved split metadata."""
    train_dataframe = load_split("train")
    val_dataframe = load_split("val")

    class_dataframe = pd.concat(
        [train_dataframe[["label", "class_name"]],
         val_dataframe[["label", "class_name"]]],
        ignore_index=True,
    )

    class_dataframe = (
        class_dataframe
        .drop_duplicates()
        .sort_values("label")
        .reset_index(drop=True)
    )

    return class_dataframe["class_name"].tolist()


def get_true_label_if_available(image_path: Path) -> str | None:
    """Return the true label if the image belongs to one saved split."""
    all_splits = pd.concat(
        [
            load_split("train"),
            load_split("val"),
            load_split("test"),
        ],
        ignore_index=True,
    )

    matched = all_splits[
        all_splits["image_path"].astype(str) == str(image_path.resolve())
    ]
    if not matched.empty:
        return matched.iloc[0]["class_name"]

    matched_by_name = all_splits[all_splits["filename"] == image_path.name]
    if not matched_by_name.empty:
        return matched_by_name.iloc[0]["class_name"]

    return None


def predict_image(
    model,
    image_path: Path,
    class_names: list[str],
    feature_set: str = "hog_lbp",
) -> tuple[str, float, list[tuple[str, float]], np.ndarray]:
    """Extract features, predict class probabilities, and return top predictions."""
    if feature_set == "uniform_lbp":
        feature_set = "lbp"

    image_bgr, image_gray = preprocess_image(image_path)

    feature_vector = extract_features(
        image_gray,
        feature_set=feature_set,
    )

    probabilities = model.predict_proba(
        feature_vector.reshape(1, -1)
    )[0]

    predicted_label = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_label]
    confidence = float(probabilities[predicted_label])

    top_indices = np.argsort(probabilities)[::-1][:3]
    top_predictions = [
        (class_names[int(index)], float(probabilities[index]))
        for index in top_indices
    ]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    return predicted_class, confidence, top_predictions, image_rgb


def save_demo_figure(
    image_rgb: np.ndarray,
    image_path: Path,
    predicted_class: str,
    confidence: float,
    top_predictions: list[tuple[str, float]],
    true_class: str | None,
    feature_set: str = "hog_lbp",
) -> Path:
    """Save input image, final prediction, and top-3 probability chart."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    class_labels = [class_name for class_name, _ in top_predictions]
    class_probabilities = [
        probability * 100
        for _, probability in top_predictions
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        gridspec_kw={"width_ratios": [1, 1.2]},
    )

    image_axis = axes[0]
    image_axis.imshow(image_rgb)
    image_axis.axis("off")

    title_lines = [
        f"Model: {feature_set.upper()} + RBF-SVM",
        f"Predicted: {predicted_class}",
        f"Confidence: {confidence:.2%}",
    ]

    if true_class is not None:
        title_lines.insert(1, f"True: {true_class}")

    image_axis.set_title("\n".join(title_lines), fontsize=11)

    chart_axis = axes[1]
    bars = chart_axis.barh(
        class_labels[::-1],
        class_probabilities[::-1],
        color="#1f77b4",
    )
    chart_axis.set_xlim(0, 100)
    chart_axis.set_xlabel("Probability (%)")
    chart_axis.set_title("Top-3 Predictions")

    for bar in bars:
        width = bar.get_width()
        chart_axis.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
        )

    fig.tight_layout()
    output_path = PREDICTIONS_DIR / f"demo_{image_path.stem}.png"
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path


def run_random_test_demo(
    feature_set: str = "lbp",
    num_samples: int = 10,
    seed: int = 42,
    top_k: int = 3,
) -> Path:
    """Evaluate and visualize N randomly selected test samples with Top-k ranking."""
    if feature_set == "uniform_lbp":
        feature_set = "lbp"

    test_df = load_split("test")
    class_names = get_class_names()
    model = load_final_model(feature_set=feature_set)

    sample_indices = test_df.sample(n=num_samples, random_state=seed).index.tolist()

    print("Demo split: held-out test")
    print(f"Number of random samples: {num_samples}")
    print(f"Random seed: {seed}")
    print(f"Sample indices: {sample_indices}")

    sampled_df = test_df.loc[sample_indices].reset_index(drop=True)

    rows = 2
    cols = int(np.ceil(num_samples / rows))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 4.6))
    axes = np.atleast_1d(axes).flatten()

    for idx, row in sampled_df.iterrows():
        img_path = Path(row["image_path"])
        true_name = row["class_name"]

        pred_class, conf, top_preds, img_rgb = predict_image(
            model=model,
            image_path=img_path,
            class_names=class_names,
            feature_set=feature_set,
        )

        ax = axes[idx]
        ax.imshow(img_rgb)
        ax.axis("off")

        # Determine pass/fail & status
        top_class_names = [name for name, _ in top_preds]
        if pred_class == true_name:
            status_text = "✓ PASS (Top-1)"
            status_color = "#2ca02c"  # Green
            border_color = "#2ca02c"
        elif true_name in top_class_names:
            rank = top_class_names.index(true_name) + 1
            status_text = f"✗ FAIL (True = Top-{rank})"
            status_color = "#ff7f0e"  # Orange
            border_color = "#ff7f0e"
        else:
            status_text = "✗ FAIL (Not in Top-3)"
            status_color = "#d62728"  # Red
            border_color = "#d62728"

        # Text summary
        pred_lines = [
            f"{i+1}. {name.replace('_', ' ').title()[:16]} ({p*100:.1f}%)"
            for i, (name, p) in enumerate(top_preds[:top_k])
        ]

        title = (
            f"File: {img_path.name}\n"
            f"True: {true_name.replace('_', ' ').title()[:18]}\n"
            + "\n".join(pred_lines) + "\n"
            + f"Status: {status_text}"
        )

        ax.set_title(title, fontsize=8.5, fontweight="bold", color="#111111", pad=6)

        # Draw colored border
        rect = Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            edgecolor=border_color,
            linewidth=3.5,
            clip_on=False,
        )
        ax.add_patch(rect)

    for unused_idx in range(num_samples, len(axes)):
        axes[unused_idx].axis("off")

    fig.suptitle(
        f"Random {num_samples} Held-Out Test Samples — Top-{top_k} Predictions ({feature_set.upper()} + RBF-SVM)\n"
        "[Green: Top-1 Correct | Orange: True Label in Top-3 | Red: Missed]",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "random_10_test_top3.png"
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved random 10 test figure: {out_path}")
    return out_path


def main() -> None:
    """Predict a leaf class or generate multi-sample test demonstration grid."""
    parser = argparse.ArgumentParser(
        description="Predict leaf classes with final RBF-SVM model."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to single image, e.g. data/raw/Flavia/1001.jpg",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Dataset split to evaluate randomly (e.g. test).",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=10,
        dest="random_samples",
        help="Number of random samples to visualize (default: 10).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top classes to rank (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--feature-set",
        "--final-model",
        type=str,
        default="hog_lbp",
        dest="feature_set",
        choices=["hog", "lbp", "uniform_lbp", "hog_lbp"],
        help="Feature set to use for inference (default: hog_lbp)",
    )
    args = parser.parse_args()

    feature_set = "lbp" if args.feature_set == "uniform_lbp" else args.feature_set

    if args.split == "test" or (args.image is None and args.split is not None):
        run_random_test_demo(
            feature_set=feature_set,
            num_samples=args.random_samples,
            seed=args.seed,
            top_k=args.top_k,
        )
        return

    # Single image mode
    image_path = args.image if args.image is not None else Path("data/raw/Flavia/1001.jpg")
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    model = load_final_model(feature_set=feature_set)
    class_names = get_class_names()

    predicted_class, confidence, top_predictions, image_rgb = predict_image(
        model=model,
        image_path=image_path,
        class_names=class_names,
        feature_set=feature_set,
    )

    true_class = get_true_label_if_available(image_path)

    print(f"\nFinal model: {feature_set.upper()} + RBF-SVM")
    print(f"Input image: {image_path}")
    print(f"Predicted class: {predicted_class}")
    if true_class is not None:
        print(f"True class: {true_class}")
    print(f"Confidence: {confidence:.2%}")
    print("\nTop-3 predictions:")
    for rank, (class_name, probability) in enumerate(top_predictions, start=1):
        print(f"{rank}. {class_name}: {probability:.2%}")

    output_path = save_demo_figure(
        image_rgb=image_rgb,
        image_path=image_path,
        predicted_class=predicted_class,
        confidence=confidence,
        top_predictions=top_predictions,
        true_class=true_class,
        feature_set=feature_set,
    )
    print(f"\nSaved demo figure: {output_path}")


if __name__ == "__main__":
    main()