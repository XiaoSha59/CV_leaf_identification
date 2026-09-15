import argparse
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, PREDICTIONS_DIR
from src.data_loader import load_split
from src.features import extract_features
from src.preprocess import preprocess_image


def load_final_model(feature_set: str = "hog_lbp"):
    """Load the trained RBF-SVM pipeline for the requested feature set."""
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


def main() -> None:
    """Run inference on a single image and save a visualization chart."""
    parser = argparse.ArgumentParser(
        description="Predict a leaf class with the final RBF-SVM model."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to an image, e.g. data/raw/Flavia/1001.jpg",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="hog_lbp",
        choices=["hog", "lbp", "hog_lbp"],
        help="Feature set to use for inference (default: hog_lbp)",
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    feature_set = args.feature_set
    model = load_final_model(feature_set=feature_set)
    class_names = get_class_names()

    predicted_class, confidence, top_predictions, image_rgb = predict_image(
        model=model,
        image_path=args.image,
        class_names=class_names,
        feature_set=feature_set,
    )

    true_class = get_true_label_if_available(args.image)

    print(f"\nFinal model: {feature_set.upper()} + RBF-SVM")
    print(f"Input image: {args.image}")
    print(f"Predicted class: {predicted_class}")

    if true_class is not None:
        print(f"True class: {true_class}")

    print(f"Confidence: {confidence:.2%}")
    print("\nTop-3 predictions:")

    for rank, (class_name, probability) in enumerate(
        top_predictions,
        start=1,
    ):
        print(f"{rank}. {class_name}: {probability:.2%}")

    output_path = save_demo_figure(
        image_rgb=image_rgb,
        image_path=args.image,
        predicted_class=predicted_class,
        confidence=confidence,
        top_predictions=top_predictions,
        true_class=true_class,
        feature_set=feature_set,
    )

    print(f"\nSaved demo figure: {output_path}")


if __name__ == "__main__":
    main()