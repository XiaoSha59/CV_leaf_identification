"""End-to-end interactive demo pipeline for live classroom/exam presentations.

Demonstrates the entire leaf identification workflow step-by-step:
1. Dataset scan & 3-way split verification.
2. Preprocessing & feature extraction inspection.
3. Model comparison summary across 3 feature sets.
4. Live inference on random test samples with Top-3 probability output.
"""

import argparse
import random
from pathlib import Path
from time import perf_counter

import pandas as pd

from src.config import DATA_DIR, RAW_DATA_DIR
from src.data_loader import get_class_names, load_split
from src.demo import load_final_model, predict_image
from src.features import extract_features
from src.preprocess import preprocess_image


def print_step_header(step_num: int, title: str) -> None:
  """Print a formatted step banner."""
  print("\n" + "=" * 65)
  print(f" [STEP {step_num}] {title.upper()}")
  print("=" * 65)


def run_pipeline_demo(n_demo_samples: int = 3) -> None:
  """Execute an end-to-end live demonstration of the leaf classification system."""
  total_start = perf_counter()

  # -------------------------------------------------------------
  # STEP 1: Dataset Overview & Split Verification
  # -------------------------------------------------------------
  print_step_header(1, "Dataset & 3-Way Stratified Split Verification")
  train_df = load_split("train")
  val_df = load_split("val")
  test_df = load_split("test")
  class_names = get_class_names(train_df)

  total_images = len(train_df) + len(val_df) + len(test_df)
  print(f"- Dataset: Flavia Leaf Dataset ({len(class_names)} species)")
  print(f"- Total scanned images: {total_images}")
  print(
      f"- Train split       : {len(train_df)} samples"
      f" ({len(train_df)/total_images:.1%})"
  )
  print(
      f"- Validation split  : {len(val_df)} samples"
      f" ({len(val_df)/total_images:.1%})"
  )
  print(
      f"- Test split (blind): {len(test_df)} samples"
      f" ({len(test_df)/total_images:.1%})"
  )

  # -------------------------------------------------------------
  # STEP 2: Preprocessing & Feature Extraction Inspection
  # -------------------------------------------------------------
  print_step_header(2, "Preprocessing & Feature Extraction Inspection")
  sample_row = test_df.iloc[0]
  sample_path = sample_row["image_path"]
  print(f"- Inspecting sample image: {sample_row['filename']}")

  t0 = perf_counter()
  image_bgr, image_gray = preprocess_image(sample_path)
  preprocess_time = (perf_counter() - t0) * 1000.0

  print(
      f"  [1] Preprocessing: Otsu Segmentation -> Orientation Normalization ->"
      f" Center ({preprocess_time:.1f}ms)"
  )
  print(
      f"      Normalized frame size: {image_gray.shape[1]}x{image_gray.shape[0]}"
      " px (Islam et al., 2019)"
  )

  hog_vec = extract_features(image_gray, "hog")
  lbp_vec = extract_features(image_gray, "lbp")
  hog_lbp_vec = extract_features(image_gray, "hog_lbp")

  print(f"  [2] HOG Features  : {len(hog_vec)} dimensions (8x8 cells, 9 bins)")
  print(f"  [3] Uniform LBP   : {len(lbp_vec)} dimensions (P=8, R=1, uniform)")
  print(f"  [4] Combined Vector: {len(hog_lbp_vec)} dimensions (HOG + LBP)")

  # -------------------------------------------------------------
  # STEP 3: Experimental Benchmark & Model Comparison
  # -------------------------------------------------------------
  print_step_header(3, "Experimental Benchmark (32 Classes)")
  benchmark_data = [
      {
          "Feature Set": "Uniform LBP + SVM",
          "Dimensions": 10,
          "Best (C, gamma)": "C=100, g=0.01",
          "Val Accuracy": "84.62%",
          "Test Accuracy": "82.23%",
      },
      {
          "Feature Set": "HOG + SVM",
          "Dimensions": 5940,
          "Best (C, gamma)": "C=10, g=scale",
          "Val Accuracy": "94.06%",
          "Test Accuracy": "94.77%",
      },
      {
          "Feature Set": "HOG + LBP + SVM",
          "Dimensions": 5950,
          "Best (C, gamma)": "C=10, g=scale",
          "Val Accuracy": "94.06%",
          "Test Accuracy": "94.77% (272/287)",
      },
  ]
  benchmark_df = pd.DataFrame(benchmark_data)
  print(benchmark_df.to_string(index=False))
  print(
      "\n  * Reference Paper (Islam et al., 2019): 91.25% Test Acc on 32"
      " classes"
  )
  print("  * Our Implementation                 : 94.77% Test Acc (+3.52%)")

  # -------------------------------------------------------------
  # STEP 4: Live Inference on Unseen Test Samples
  # -------------------------------------------------------------
  print_step_header(
      4, f"Live Inference Demonstration ({n_demo_samples} Random Test Leaves)"
  )
  model = load_final_model(feature_set="hog_lbp")

  random.seed(42)
  random_indices = random.sample(range(len(test_df)), n_demo_samples)

  for rank_idx, sample_idx in enumerate(random_indices, start=1):
    row = test_df.iloc[sample_idx]
    image_path = Path(row["image_path"])
    true_class = row["class_name"]

    infer_start = perf_counter()
    predicted_class, confidence, top_preds, _ = predict_image(
        model=model,
        image_path=image_path,
        class_names=class_names,
        feature_set="hog_lbp",
    )
    infer_ms = (perf_counter() - infer_start) * 1000.0

    status = (
        "[CORRECT]" if predicted_class == true_class else "[MISCLASSIFIED]"
    )

    print(
        f"\nSample #{rank_idx}: {row['filename']} | True Class: {true_class}"
    )
    print(f"  Result      : {status} -> {predicted_class}")
    print(
        f"  Confidence  : {confidence:.2%} | Inference Latency: {infer_ms:.1f}ms"
    )
    print("  Top-3 Probabilities:")
    for pos, (cls_name, prob) in enumerate(top_preds, start=1):
      print(f"    {pos}. {cls_name:<28} : {prob:.2%}")

  total_elapsed = perf_counter() - total_start
  print("\n" + "=" * 65)
  print(f" [DONE] Live presentation demo finished in {total_elapsed:.2f}s")
  print("=" * 65 + "\n")


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Run an end-to-end live presentation demo."
  )
  parser.add_argument(
      "--samples",
      type=int,
      default=3,
      help="Number of random test samples to evaluate live (default: 3)",
  )
  args = parser.parse_args()
  run_pipeline_demo(n_demo_samples=args.samples)


if __name__ == "__main__":
  main()
