import argparse
import json
from pathlib import Path
from time import perf_counter
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

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
    PCA_HOG_COMPONENTS,
    PREDICTIONS_DIR,
    RANDOM_STATE,
)
from src.data_loader import (
    get_class_names,
    load_split,
    verify_image_paths,
)
from src.features import extract_features, extract_hog, extract_lbp_histogram
from src.fusion import fuse_hog_pca_lbp
from src.pca_reduction import fit_pca, save_pca
from src.preprocess import preprocess_image


def build_feature_matrix(
    dataframe: pd.DataFrame,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray]:
  """Extract feature matrix and labels for a dataset split."""
  feature_vectors = []
  total_samples = len(dataframe)

  for position, (_, row) in enumerate(dataframe.iterrows(), start=1):
    _, image_gray = preprocess_image(row["image_path"])
    feature_vector = extract_features(image_gray, feature_set)
    feature_vectors.append(feature_vector)

    if position % 100 == 0 or position == total_samples:
      print(
          f"Extracted {position}/{total_samples} "
          f"{feature_set} feature vectors."
      )

  X = np.vstack(feature_vectors)
  y = dataframe["label"].to_numpy(dtype=np.int64)

  return X, y


def build_raw_representations(
    dataframe: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Extract raw HOG and LBP matrices for PCA fusion."""
  hog_vectors = []
  lbp_vectors = []
  total_samples = len(dataframe)

  for position, (_, row) in enumerate(dataframe.iterrows(), start=1):
    _, image_gray = preprocess_image(row["image_path"])
    hog_vectors.append(extract_hog(image_gray))
    lbp_vectors.append(extract_lbp_histogram(image_gray))

    if position % 100 == 0 or position == total_samples:
      print(f"Extracted {position}/{total_samples} HOG+LBP raw feature vectors.")

  X_hog = np.vstack(hog_vectors)
  X_lbp = np.vstack(lbp_vectors)
  y = dataframe["label"].to_numpy(dtype=np.int64)

  return X_hog, X_lbp, y


def create_final_model(C: float, gamma: str | float) -> Pipeline:
  """Create final StandardScaler + RBF-SVM pipeline with probability estimation."""
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
) -> Path:
  """Generate and save the normalized confusion matrix heatmap."""
  FIGURES_DIR.mkdir(parents=True, exist_ok=True)
  matrix = confusion_matrix(
      y_true,
      y_pred,
      labels=np.arange(len(class_names)),
  )

  fig, ax = plt.subplots(figsize=(18, 15))
  sns.heatmap(
      matrix,
      annot=True,
      fmt="d",
      cmap="Greens",
      xticklabels=class_names,
      yticklabels=class_names,
      cbar_kws={"label": "Sample Count"},
      ax=ax,
  )

  ax.set_title(
      f"Confusion Matrix: {feature_set.upper()} (Test Set)",
      fontsize=16,
      fontweight="bold",
      pad=15,
  )
  ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
  ax.set_ylabel("True Label", fontsize=12, labelpad=10)
  plt.xticks(rotation=45, ha="right", fontsize=9)
  plt.yticks(rotation=0, fontsize=9)

  fig.tight_layout()
  output_path = FIGURES_DIR / f"{feature_set}_test_confusion_matrix.png"
  fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
  plt.close(fig)
  print(f"Saved confusion matrix: {output_path}")
  return output_path


def save_error_cases(
    test_dataframe: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    feature_set: str,
    max_cases: int = 12,
) -> Path | None:
  """Visualize incorrectly predicted leaf images alongside top-3 predictions."""
  FIGURES_DIR.mkdir(parents=True, exist_ok=True)
  error_mask = y_true != y_pred
  error_indices = np.where(error_mask)[0]

  if len(error_indices) == 0:
    print("Zero error cases detected on the test set. Skipping error gallery.")
    return None

  total_errors = len(error_indices)
  display_count = min(total_errors, max_cases)
  selected_indices = error_indices[:display_count]

  cols = 4
  rows = int(np.ceil(display_count / cols))
  fig, axes = plt.subplots(rows, cols, figsize=(18, 4.5 * rows))
  axes = np.array(axes).flatten()

  for plot_index, error_idx in enumerate(selected_indices):
    ax = axes[plot_index]
    row = test_dataframe.iloc[error_idx]

    image_bgr, _ = preprocess_image(row["image_path"])
    image_rgb = cv2_bgr_to_rgb(image_bgr)

    true_name = class_names[y_true[error_idx]]
    pred_name = class_names[y_pred[error_idx]]

    probs = probabilities[error_idx]
    top3_indices = np.argsort(probs)[::-1][:3]
    top3_text = "\n".join(
        [f"{class_names[idx]}: {probs[idx]:.2f}" for idx in top3_indices]
    )

    ax.imshow(image_rgb)
    ax.set_title(
        f"File: {Path(row['image_path']).name}\n"
        f"True: {true_name}\n"
        f"Pred: {pred_name}\n"
        f"Top 3:\n{top3_text}",
        fontsize=9,
        color="darkred",
        fontweight="bold",
    )
    ax.axis("off")

  for unused_index in range(display_count, len(axes)):
    fig.delaxes(axes[unused_index])

  fig.suptitle(
      f"Misclassified Test Samples - {feature_set.upper()} ({total_errors} errors total)",
      fontsize=16,
      fontweight="bold",
      y=0.99,
  )
  fig.tight_layout()

  output_path = FIGURES_DIR / f"{feature_set}_error_cases.png"
  fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
  plt.close(fig)
  print(f"Saved error cases figure: {output_path}")
  return output_path


def cv2_bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
  """Convert OpenCV BGR image array to RGB for matplotlib."""
  import cv2
  return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_prediction_tables(
    test_dataframe: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> tuple[Path, Path]:
  """Save CSV files containing predictions and per-class performance metrics."""
  PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

  preds_df = test_dataframe.copy()
  preds_df["true_class"] = [class_names[label] for label in y_true]
  preds_df["predicted_class"] = [class_names[label] for label in y_pred]
  preds_df["correct"] = y_true == y_pred
  preds_df["confidence"] = np.max(probabilities, axis=1)

  detailed_path = PREDICTIONS_DIR / "test_predictions_detailed.csv"
  preds_df.to_csv(detailed_path, index=False)

  report_dict = classification_report(
      y_true,
      y_pred,
      labels=np.arange(len(class_names)),
      target_names=class_names,
      output_dict=True,
      zero_division=0,
  )
  class_report_df = pd.DataFrame(report_dict).transpose().reset_index()
  class_report_df.rename(columns={"index": "class_or_metric"}, inplace=True)

  summary_path = PREDICTIONS_DIR / "per_class_performance.csv"
  class_report_df.to_csv(summary_path, index=False)

  print(f"Saved predictions table: {detailed_path}")
  print(f"Saved per-class performance summary: {summary_path}")
  return detailed_path, summary_path


def save_metrics(
    test_accuracy: float,
    test_macro_f1: float,
    report_text: str,
    train_seconds: float,
    feature_dimension: int,
    n_train_val: int,
    n_test: int,
    feature_set: str,
) -> Path:
  """Persist final evaluation metrics as a structured JSON artifact."""
  METRICS_DIR.mkdir(parents=True, exist_ok=True)
  metrics_path = METRICS_DIR / f"{feature_set}_test_metrics.json"

  payload = {
      "feature_set": feature_set,
      "test_accuracy": float(test_accuracy),
      "test_macro_f1": float(test_macro_f1),
      "feature_dimension": feature_dimension,
      "train_val_samples": n_train_val,
      "test_samples": n_test,
      "train_time_seconds": float(train_seconds),
      "classification_report": report_text,
  }

  with metrics_path.open("w", encoding="utf-8") as file:
    json.dump(payload, file, indent=2)

  print(f"Saved final test metrics: {metrics_path}")
  return metrics_path


def main() -> None:
  """Fit final SVM on Train+Val and evaluate on Test set."""
  parser = argparse.ArgumentParser(
      description="Evaluate leaf classification on test set."
  )
  parser.add_argument(
      "--feature-set",
      "--final-model",
      type=str,
      default="hog_pca_lbp",
      dest="feature_set",
      choices=["hog", "lbp", "uniform_lbp", "hog_lbp", "hog_pca_lbp"],
      help="Feature set to evaluate (default: hog_pca_lbp)",
  )
  parser.add_argument(
      "--pca-k",
      type=int,
      default=PCA_HOG_COMPONENTS,
      help="Number of PCA components for HOG reduction (default: 64).",
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

  if feature_set == "hog_pca_lbp":
    print("\n--- Extracting raw HOG and LBP representations ---")
    X_hog_tv, X_lbp_tv, y_train_val = build_raw_representations(train_val_dataframe)
    X_hog_te, X_lbp_te, y_test = build_raw_representations(test_dataframe)

    print(f"\n--- Fitting PCA (k={args.pca_k}) on Train+Val HOG features ---")
    pca_model = fit_pca(X_hog_tv, n_components=args.pca_k)
    save_pca(pca_model, MODELS_DIR / f"final_hog_pca_{args.pca_k}.joblib")

    print("--- Fusing PCA-reduced HOG + LBP ---")
    X_train_val = fuse_hog_pca_lbp(X_hog_tv, X_lbp_tv, pca_model)
    X_test = fuse_hog_pca_lbp(X_hog_te, X_lbp_te, pca_model)
  else:
    X_train_val, y_train_val = build_feature_matrix(train_val_dataframe, feature_set)
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


if __name__ == "__main__":
  main()