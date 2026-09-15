import json
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
)
from src.data_loader import (
    get_class_names,
    load_split,
    verify_image_paths,
)
from src.features import extract_features
from src.preprocess import preprocess_image


FINAL_FEATURE_SET = "lbp"
FINAL_C = 100
FINAL_GAMMA = 0.01


def build_feature_matrix(
    dataframe: pd.DataFrame,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract handcrafted features and labels from a dataframe."""
    feature_vectors = []

    for position, (_, row) in enumerate(dataframe.iterrows(), start=1):
        _, image_gray = preprocess_image(row["image_path"])
        feature_vector = extract_features(image_gray, feature_set)
        feature_vectors.append(feature_vector)

        if position % 50 == 0 or position == len(dataframe):
            print(
                f"Extracted {position}/{len(dataframe)} "
                f"{feature_set} feature vectors."
            )

    X = np.vstack(feature_vectors)
    y = dataframe["label"].to_numpy(dtype=np.int64)

    return X, y


def create_final_model() -> Pipeline:
    """Create the final scaled RBF-SVM using validation-selected parameters."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    kernel="rbf",
                    C=FINAL_C,
                    gamma=FINAL_GAMMA,
                    probability=True,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> None:
    """Save normalized test confusion matrix."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        normalize="true",
    )

    fig, axis = plt.subplots(figsize=(11, 9))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Recall per true class"},
        ax=axis,
    )

    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Normalized Confusion Matrix: Final LBP + RBF-SVM")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    fig.tight_layout()
    output_path = FIGURES_DIR / "confusion_matrix_final_lbp.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved confusion matrix: {output_path}")


def save_error_cases(
    test_dataframe: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    max_examples: int = 9,
) -> None:
    """Save a grid of incorrectly classified test images."""
    error_indices = np.flatnonzero(y_true != y_pred)

    if len(error_indices) == 0:
        print("No misclassified test images; error case figure was not created.")
        return

    selected_indices = error_indices[:max_examples]
    columns = 3
    rows = int(np.ceil(len(selected_indices) / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 4 * rows),
    )
    axes = np.atleast_1d(axes).ravel()

    for axis, sample_index in zip(axes, selected_indices):
        image_bgr, _ = preprocess_image(
            test_dataframe.iloc[sample_index]["image_path"]
        )
        image_rgb = image_bgr[:, :, ::-1]

        true_label = int(y_true[sample_index])
        predicted_label = int(y_pred[sample_index])
        confidence = float(probabilities[sample_index, predicted_label])

        axis.imshow(image_rgb)
        axis.set_title(
            f"True: {class_names[true_label]}\n"
            f"Pred: {class_names[predicted_label]}\n"
            f"Probability: {confidence:.2f}",
            fontsize=9,
        )
        axis.axis("off")

    for axis in axes[len(selected_indices):]:
        axis.axis("off")

    fig.suptitle(
        "Misclassified Test Images: Final LBP + RBF-SVM",
        fontsize=15,
    )
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / "error_cases_final_lbp.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved error cases: {output_path}")


def save_metrics(
    test_accuracy: float,
    test_macro_f1: float,
    report_text: str,
    train_seconds: float,
    feature_dimension: int,
    n_train_val: int,
    n_test: int,
) -> None:
  """Save final test metrics and classification report."""
  METRICS_DIR.mkdir(parents=True, exist_ok=True)

  metrics = {
      "final_feature_set": FINAL_FEATURE_SET,
      "C": FINAL_C,
      "gamma": FINAL_GAMMA,
      "feature_dimension": feature_dimension,
      "train_seconds": train_seconds,
      "test_accuracy": test_accuracy,
      "test_macro_f1": test_macro_f1,
      "n_train_plus_val": n_train_val,
      "n_test": n_test,
  }

  metrics_path = METRICS_DIR / "final_test_metrics.json"
  report_path = METRICS_DIR / "final_classification_report.txt"

  with metrics_path.open("w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

  with report_path.open("w", encoding="utf-8") as file:
    file.write(report_text)

  print(f"Saved test metrics: {metrics_path}")
  print(f"Saved classification report: {report_path}")

def main() -> None:
    train_dataframe = load_split("train")
    val_dataframe = load_split("val")
    test_dataframe = load_split("test")

    train_val_dataframe = pd.concat(
        [train_dataframe, val_dataframe],
        ignore_index=True,
    )

    verify_image_paths(train_val_dataframe)
    verify_image_paths(test_dataframe)

    class_names = get_class_names(train_val_dataframe)

    print(
        "Final model setup:\n"
        f"Feature set: {FINAL_FEATURE_SET}\n"
        f"C: {FINAL_C}\n"
        f"Gamma: {FINAL_GAMMA}\n"
        f"Train + validation samples: {len(train_val_dataframe)}\n"
        f"Test samples: {len(test_dataframe)}"
    )

    X_train_val, y_train_val = build_feature_matrix(
        train_val_dataframe,
        FINAL_FEATURE_SET,
    )
    X_test, y_test = build_feature_matrix(
        test_dataframe,
        FINAL_FEATURE_SET,
    )

    print(
        f"\nFeature matrix shapes:\n"
        f"X_train_val: {X_train_val.shape}\n"
        f"X_test: {X_test.shape}"
    )

    final_model = create_final_model()

    start_time = perf_counter()
    final_model.fit(X_train_val, y_train_val)
    train_seconds = perf_counter() - start_time

    y_test_pred = final_model.predict(X_test)
    test_probabilities = final_model.predict_proba(X_test)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_macro_f1 = f1_score(
        y_test,
        y_test_pred,
        average="macro",
        zero_division=0,
    )

    report_text = classification_report(
        y_test,
        y_test_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print("\nFinal test results:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Macro F1: {test_macro_f1:.4f}")
    print("\nClassification report:")
    print(report_text)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "final_lbp_svm.joblib"
    joblib.dump(final_model, model_path)
    print(f"Saved final model: {model_path}")

    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_test_pred,
        class_names=class_names,
    )

    save_error_cases(
        test_dataframe=test_dataframe,
        y_true=y_test,
        y_pred=y_test_pred,
        probabilities=test_probabilities,
        class_names=class_names,
    )

    save_metrics(
        test_accuracy=test_accuracy,
        test_macro_f1=test_macro_f1,
        report_text=report_text,
        train_seconds=train_seconds,
        feature_dimension=X_train_val.shape[1],
        n_train_val=len(train_val_dataframe),
        n_test=len(test_dataframe),
    )

if __name__ == "__main__":
    main()