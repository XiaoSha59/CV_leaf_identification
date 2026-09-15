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


FINAL_MODEL_PATH = MODELS_DIR / "final_lbp_svm.joblib"
FINAL_FEATURE_SET = "lbp"


def load_final_model():
    """Load the final validation-selected LBP + RBF-SVM pipeline."""
    if not FINAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Final model not found: {FINAL_MODEL_PATH}\n"
            "Run `python -m src.evaluate` first."
        )

    return joblib.load(FINAL_MODEL_PATH)


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

    resolved_input_path = image_path.resolve()

    for _, row in all_splits.iterrows():
        saved_path = Path(row["image_path"]).resolve()

        if saved_path == resolved_input_path:
            return str(row["class_name"])

    return None


def predict_image(
    model,
    image_path: Path,
    class_names: list[str],
) -> tuple[str, float, list[tuple[str, float]], np.ndarray]:
    """Predict one image and return label, confidence, top-3, and original RGB."""
    image_bgr, image_gray = preprocess_image(image_path)

    feature_vector = extract_features(
        image_gray,
        feature_set=FINAL_FEATURE_SET,
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
        f"Predicted: {predicted_class}",
        f"Confidence: {confidence:.2%}",
    ]

    if true_class is not None:
        title_lines.insert(0, f"True: {true_class}")

    image_axis.set_title("\n".join(title_lines), fontsize=12)

    probability_axis = axes[1]
    colors = ["#59A14F", "#4C78A8", "#B0B0B0"]

    probability_axis.barh(
        class_labels[::-1],
        class_probabilities[::-1],
        color=colors[::-1],
    )
    probability_axis.set_xlim(0, 100)
    probability_axis.set_xlabel("Probability (%)")
    probability_axis.set_title("Top-3 Predictions")
    probability_axis.grid(axis="x", alpha=0.25)

    for index, value in enumerate(class_probabilities[::-1]):
        probability_axis.text(
            value + 1,
            index,
            f"{value:.2f}%",
            va="center",
            fontsize=10,
        )

    fig.suptitle(
        f"Leaf Classification Demo: {image_path.name}",
        fontsize=14,
    )
    fig.tight_layout()

    output_path = PREDICTIONS_DIR / f"demo_{image_path.stem}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict a leaf class with the final Uniform LBP + RBF-SVM model."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to an image, e.g. data/raw/Flavia/1001.jpg",
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    model = load_final_model()
    class_names = get_class_names()

    predicted_class, confidence, top_predictions, image_rgb = predict_image(
        model=model,
        image_path=args.image,
        class_names=class_names,
    )

    true_class = get_true_label_if_available(args.image)

    print("\nFinal model: Uniform LBP + RBF-SVM")
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
    )

    print(f"\nSaved demo figure: {output_path}")


if __name__ == "__main__":
    main()