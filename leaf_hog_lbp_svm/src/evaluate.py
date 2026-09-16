import argparse
import json
from pathlib import Path
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
    PREDICTIONS_DIR,
    RANDOM_STATE,
)
from src.data_loader import (
    get_class_names,
    load_split,
    verify_image_paths,
)
from src.features import extract_features
from src.preprocess import preprocess_image


def build_feature_matrix(
    dataframe: pd.DataFrame,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray]:
  """Extract handcrafted features and labels from a dataset split dataframe."""
  feature_vectors = []
  total_samples = len(dataframe)

  for position, (_, row) in enumerate(dataframe.iterrows(), start=1):
    _, image_gray = preprocess_image(row["image_path"])
    feature_vector = extract_features(image_gray, feature_set)
    feature_vectors.append(feature_vector)

    if position % 50 == 0 or position == total_samples:
      print(
          f"Extracted {position}/{total_samples} "
          f"{feature_set} feature vectors."
      )

  X = np.vstack(feature_vectors)
  y = dataframe["label"].to_numpy(dtype=np.int64)

  return X, y


def create_final_model(C: float, gamma: str | float) -> Pipeline:
  """Create the final scaled RBF-SVM using validation-selected parameters."""
  return Pipeline(
      steps=[
          ("scaler", StandardScaler()),
          (
              "svm",
              SVC(
                  kernel="rbf",
                  C=C,
                  gamma=gamma,
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
    feature_set: str,
) -> None:
  """Save normalized test confusion matrix heatmap figure."""
  FIGURES_DIR.mkdir(parents=True, exist_ok=True)

  matrix = confusion_matrix(
      y_true,
      y_pred,
      labels=np.arange(len(class_names)),
      normalize="true",
  )

  num_classes = len(class_names)
  fig_size = (16, 14) if num_classes > 15 else (11, 9)
  fig, axis = plt.subplots(figsize=fig_size)

  sns.heatmap(
      matrix,
      annot=num_classes <= 32,
      fmt=".2f",
      cmap="Blues",
      xticklabels=class_names,
      yticklabels=class_names,
      annot_kws={"size": 7 if num_classes > 15 else 9},
      cbar_kws={"label": "Recall per true class"},
      ax=axis,
  )

  axis.set_xlabel("Predicted class")
  axis.set_ylabel("True class")
  axis.set_title(
      f"Normalized Confusion Matrix: Final {feature_set.upper()} + RBF-SVM"
  )
  plt.xticks(rotation=45, ha="right", fontsize=8 if num_classes > 15 else 10)
  plt.yticks(rotation=0, fontsize=8 if num_classes > 15 else 10)

  fig.tight_layout()
  output_path = FIGURES_DIR / f"confusion_matrix_final_{feature_set}.png"
  fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
  plt.close(fig)

  print(f"Saved confusion matrix: {output_path}")


def save_error_cases(
    test_dataframe: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    feature_set: str,
    max_examples: int = 9,
) -> None:
  """Save a visual grid of misclassified test images with file names and confidence."""
  error_indices = np.flatnonzero(y_true != y_pred)

  if len(error_indices) == 0:
    print("No misclassified test images; error case figure was not created.")
    return

  FIGURES_DIR.mkdir(parents=True, exist_ok=True)
  selected_indices = error_indices[:max_examples]
  n_samples = len(selected_indices)

  cols = 3
  rows = int(np.ceil(n_samples / cols))
  fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
  axes = np.atleast_1d(axes).flatten()

  for plot_idx, error_idx in enumerate(selected_indices):
    image_path = Path(test_dataframe.iloc[error_idx]["image_path"])
    true_label = y_true[error_idx]
    pred_label = y_pred[error_idx]

    true_name = class_names[true_label]
    pred_name = class_names[pred_label]
    confidence = probabilities[error_idx, pred_label] * 100.0

    image_rgb = plt.imread(str(image_path))

    axes[plot_idx].imshow(image_rgb)
    axes[plot_idx].set_title(
        f"File: {image_path.name}\n"
        f"True: {true_name}\n"
        f"Pred: {pred_name} ({confidence:.1f}%)",
        fontsize=9,
        color="darkred",
    )
    axes[plot_idx].axis("off")

  for unused_idx in range(n_samples, len(axes)):
    axes[unused_idx].axis("off")

  fig.suptitle(
      f"Misclassified Test Samples ({feature_set.upper()} + RBF-SVM)",
      fontsize=14,
  )
  fig.tight_layout()

  output_path = FIGURES_DIR / f"error_cases_final_{feature_set}.png"
  fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
  plt.close(fig)

  print(f"Saved error cases figure: {output_path}")


def save_prediction_tables(
    test_dataframe: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> None:
  """Export comprehensive test predictions and misclassified samples CSV files."""
  PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

  predicted_names = [class_names[pred] for pred in y_pred]
  confidence_scores = np.max(probabilities, axis=1)

  predictions_df = test_dataframe.copy()
  predictions_df["predicted_label"] = y_pred
  predictions_df["predicted_class"] = predicted_names
  predictions_df["confidence"] = confidence_scores
  predictions_df["is_correct"] = y_true == y_pred

  all_preds_path = PREDICTIONS_DIR / "test_predictions.csv"
  predictions_df.to_csv(all_preds_path, index=False)
  print(f"Saved all predictions: {all_preds_path}")

  misclassified_df = predictions_df[~predictions_df["is_correct"]].copy()
  misclassified_path = PREDICTIONS_DIR / "misclassified_test_samples.csv"
  misclassified_df.to_csv(misclassified_path, index=False)
  print(
      f"Saved misclassified samples table ({len(misclassified_df)} errors):"
      f" {misclassified_path}"
  )


def save_metrics(
    test_accuracy: float,
    test_macro_f1: float,
    report_text: str,
    train_seconds: float,
    feature_dimension: int,
    n_train_val: int,
    n_test: int,
    feature_set: str,
) -> None:
  """Persist test evaluation metrics to JSON and text summary files."""
  METRICS_DIR.mkdir(parents=True, exist_ok=True)

  metrics = {
      "feature_set": feature_set,
      "test_accuracy": float(test_accuracy),
      "test_macro_f1": float(test_macro_f1),
      "train_seconds": float(train_seconds),
      "feature_dimension": int(feature_dimension),
      "n_train_val_samples": int(n_train_val),
      "n_test_samples": int(n_test),
  }

  json_path = METRICS_DIR / f"final_test_metrics_{feature_set}.json"
  with open(json_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

  with open(METRICS_DIR / "final_test_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

  print(f"Saved test metrics: {json_path}")

  report_path = METRICS_DIR / "final_classification_report.txt"
  with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
  print(f"Saved classification report: {report_path}")


def save_final_test_summary_card(
    feature_set: str,
    test_accuracy: float,
    test_macro_f1: float,
    n_train_val: int,
    n_test: int,
    num_classes: int,
) -> Path:
  """Save a presentation-ready graphic summary card of held-out test evaluation."""
  FIGURES_DIR.mkdir(parents=True, exist_ok=True)

  feature_label_map = {
      "hog": "HOG + RBF-SVM",
      "lbp": "Uniform LBP + RBF-SVM",
      "uniform_lbp": "Uniform LBP + RBF-SVM",
      "hog_lbp": "HOG + Uniform LBP + RBF-SVM",
  }
  model_name = feature_label_map.get(feature_set, feature_set)

  fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="#1e1e2e")
  ax.set_facecolor("#1e1e2e")
  ax.axis("off")

  card_text = (
      "FINAL HELD-OUT TEST EVALUATION\n"
      "─────────────────────────────────────────────\n"
      f"Selected Model   : {model_name}\n"
      "Selection Source : Validation Macro F1\n"
      f"Refit Data       : Train + Validation ({n_train_val} samples)\n"
      f"Test Set         : {n_test} samples ({num_classes} classes)\n"
      "─────────────────────────────────────────────\n"
      f"Test Accuracy    : {test_accuracy * 100.0:.2f}%\n"
      f"Test Macro F1    : {test_macro_f1 * 100.0:.2f}%\n"
      "─────────────────────────────────────────────"
  )

  ax.text(
      0.08,
      0.88,
      card_text,
      fontsize=13,
      fontfamily="monospace",
      color="#cdd6f4",
      va="top",
      ha="left",
      linespacing=1.6,
  )

  output_path = FIGURES_DIR / "final_test_summary.png"
  fig.savefig(str(output_path), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
  plt.close(fig)
  print(f"Saved final test summary card: {output_path}")
  return output_path


def main() -> None:
  """Fit final SVM on Train+Val and evaluate on Test set."""
  parser = argparse.ArgumentParser(
      description="Evaluate leaf classification on test set."
  )
  parser.add_argument(
      "--feature-set",
      "--final-model",
      type=str,
      default="hog_lbp",
      dest="feature_set",
      choices=["hog", "lbp", "uniform_lbp", "hog_lbp"],
      help="Feature set or final model to evaluate (default: hog_lbp)",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed (optional).",
  )
  args = parser.parse_args()

  feature_set = "lbp" if args.feature_set == "uniform_lbp" else args.feature_set

  val_metrics_path = METRICS_DIR / f"{feature_set}_validation_metrics.json"
  if val_metrics_path.exists():
    with open(val_metrics_path, "r", encoding="utf-8") as f:
      val_metrics = json.load(f)
    if "best_validation_result" in val_metrics:
      best_c = val_metrics["best_validation_result"]["C"]
      best_gamma = val_metrics["best_validation_result"]["gamma"]
    else:
      best_c = val_metrics.get("C", 10)
      best_gamma = val_metrics.get("gamma", "scale")
  else:
    best_c = 10
    best_gamma = "scale"

  train_dataframe = load_split("train")
  val_dataframe = load_split("val")
  test_dataframe = load_split("test")

  verify_image_paths(train_dataframe)
  verify_image_paths(val_dataframe)
  verify_image_paths(test_dataframe)

  class_names = get_class_names(train_dataframe)

  train_val_dataframe = (
      pd.concat([train_dataframe, val_dataframe], ignore_index=True)
      .sample(frac=1.0, random_state=args.seed)
      .reset_index(drop=True)
  )

  print("Final model setup:")
  print(f"Feature set: {feature_set}")
  print(f"C: {best_c}")
  print(f"Gamma: {best_gamma}")
  print(f"Train + validation samples: {len(train_val_dataframe)}")
  print(f"Test samples: {len(test_dataframe)}")

  X_train_val, y_train_val = build_feature_matrix(
      train_val_dataframe, feature_set
  )
  X_test, y_test = build_feature_matrix(test_dataframe, feature_set)

  print("\nFeature matrix shapes:")
  print(f"X_train_val: {X_train_val.shape}")
  print(f"X_test: {X_test.shape}")

  final_model = create_final_model(C=best_c, gamma=best_gamma)

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
  model_path = MODELS_DIR / f"final_{feature_set}_svm.joblib"
  joblib.dump(final_model, model_path)
  print(f"Saved final model: {model_path}")

  save_confusion_matrix(
      y_true=y_test,
      y_pred=y_test_pred,
      class_names=class_names,
      feature_set=feature_set,
  )

  save_error_cases(
      test_dataframe=test_dataframe,
      y_true=y_test,
      y_pred=y_test_pred,
      probabilities=test_probabilities,
      class_names=class_names,
      feature_set=feature_set,
  )

  save_prediction_tables(
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
      feature_set=feature_set,
  )

  save_final_test_summary_card(
      feature_set=feature_set,
      test_accuracy=test_accuracy,
      test_macro_f1=test_macro_f1,
      n_train_val=len(train_val_dataframe),
      n_test=len(test_dataframe),
      num_classes=len(class_names),
  )


if __name__ == "__main__":
  main()