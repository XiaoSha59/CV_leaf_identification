import argparse
import json
from itertools import product
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import METRICS_DIR, MODELS_DIR, RANDOM_STATE
from src.data_loader import load_split, verify_image_paths
from src.features import extract_features
from src.preprocess import preprocess_image


def build_feature_matrix(
    dataframe: pd.DataFrame,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray]:
  """Extract a feature matrix X and label array y for a given dataset split."""
  feature_vectors = []
  total_samples = len(dataframe)

  for index, row in dataframe.iterrows():
    _, image_gray = preprocess_image(row["image_path"])
    feature_vector = extract_features(image_gray, feature_set)
    feature_vectors.append(feature_vector)

    if (index + 1) % 50 == 0 or (index + 1) == total_samples:
      print(f"Extracted {index + 1}/{total_samples} {feature_set} feature vectors.")

  X = np.vstack(feature_vectors)
  y = dataframe["label"].to_numpy(dtype=np.int64)

  return X, y


def create_svm_pipeline(C: float, gamma: str | float) -> Pipeline:
  """Construct a machine learning pipeline with feature scaling and RBF-SVM."""
  return Pipeline(
      steps=[
          ("scaler", StandardScaler()),
          (
              "svm",
              SVC(
                  kernel="rbf",
                  C=C,
                  gamma=gamma,
                  random_state=RANDOM_STATE,
              ),
          ),
      ]
  )


def tune_on_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[Pipeline, dict]:
  """Evaluate candidate hyperparameter configurations on the validation set."""
  C_values = [0.1, 1, 10, 100]
  gamma_values: list[str | float] = ["scale", 0.001, 0.01]

  best_model = None
  best_result = None

  for C, gamma in product(C_values, gamma_values):
    model = create_svm_pipeline(C=C, gamma=gamma)

    start_time = perf_counter()
    model.fit(X_train, y_train)
    train_seconds = perf_counter() - start_time

    y_val_pred = model.predict(X_val)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_macro_f1 = f1_score(
        y_val,
        y_val_pred,
        average="macro",
        zero_division=0,
    )

    result = {
        "C": C,
        "gamma": gamma,
        "val_accuracy": float(val_accuracy),
        "val_macro_f1": float(val_macro_f1),
        "train_seconds": float(train_seconds),
    }

    print(
        f"C={C:<5} gamma={str(gamma):<5} | "
        f"val_accuracy={val_accuracy:.4f} | "
        f"val_macro_f1={val_macro_f1:.4f} | "
        f"train_seconds={train_seconds:.2f}s"
    )

    if best_result is None or val_macro_f1 > best_result["val_macro_f1"]:
      best_model = model
      best_result = result

  if best_model is None or best_result is None:
    raise RuntimeError("Hyperparameter tuning failed to produce a valid model.")

  return best_model, best_result


def save_training_artifacts(
    model: Pipeline,
    best_result: dict,
    feature_set: str,
    feature_dimension: int,
) -> None:
  """Persist the trained pipeline and validation performance metrics to disk."""
  MODELS_DIR.mkdir(parents=True, exist_ok=True)
  METRICS_DIR.mkdir(parents=True, exist_ok=True)

  model_path = MODELS_DIR / f"{feature_set}_svm.joblib"
  metrics_path = METRICS_DIR / f"{feature_set}_validation_metrics.json"

  joblib.dump(model, model_path)

  output = {
      "feature_set": feature_set,
      "feature_dimension": feature_dimension,
      "best_validation_result": best_result,
  }

  with metrics_path.open("w", encoding="utf-8") as file:
    json.dump(output, file, indent=2)

  print(f"\nSaved model to: {model_path}")
  print(f"Saved validation metrics to: {metrics_path}")


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Train and tune an RBF-SVM model on a specified feature set."
  )
  parser.add_argument(
      "--feature-set",
      choices=["hog", "lbp", "hog_lbp"],
      required=True,
      help="Handcrafted feature representation to extract and train.",
  )
  args = parser.parse_args()

  train_dataframe = load_split("train")
  val_dataframe = load_split("val")

  verify_image_paths(train_dataframe)
  verify_image_paths(val_dataframe)

  print(
      f"Training feature set: {args.feature_set}\n"
      f"Train samples: {len(train_dataframe)}\n"
      f"Validation samples: {len(val_dataframe)}"
  )

  X_train, y_train = build_feature_matrix(
      train_dataframe,
      args.feature_set,
  )
  X_val, y_val = build_feature_matrix(
      val_dataframe,
      args.feature_set,
  )

  print(
      f"\nFeature matrix shapes:\n"
      f"X_train: {X_train.shape}\n"
      f"X_val:   {X_val.shape}\n"
  )

  best_model, best_result = tune_on_validation(
      X_train,
      y_train,
      X_val,
      y_val,
  )

  print("\nBest validation configuration:")
  print(json.dumps(best_result, indent=2))

  save_training_artifacts(
      model=best_model,
      best_result=best_result,
      feature_set=args.feature_set,
      feature_dimension=X_train.shape[1],
  )


if __name__ == "__main__":
  main()