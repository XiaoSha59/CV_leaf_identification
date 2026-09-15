import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    LABELS_PATH,
    RANDOM_STATE,
    TEST_RATIO,
    TEST_SPLIT_PATH,
    TRAIN_RATIO,
    TRAIN_SPLIT_PATH,
    VAL_RATIO,
    VAL_SPLIT_PATH,
)


def load_labels() -> pd.DataFrame:
  """Load labels dataframe from the generated CSV file."""
  if not LABELS_PATH.exists():
    raise FileNotFoundError(
        f"Labels file not found at: {LABELS_PATH}\n"
        "Please run `python -m src.make_labels` first."
    )
  return pd.read_csv(LABELS_PATH)


def split_train_val_test(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Split dataset into Train, Validation, and Test subsets with stratification."""
  if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-6:
    raise ValueError("Split ratios must sum to 1.0.")

  temp_ratio = VAL_RATIO + TEST_RATIO

  # Step 1: Split into Train and temporary (Val + Test) sets
  train_df, temp_df = train_test_split(
      df,
      test_size=temp_ratio,
      random_state=RANDOM_STATE,
      stratify=df["label"],
  )

  # Step 2: Split temporary set into Validation and Test sets
  test_fraction = TEST_RATIO / temp_ratio
  val_df, test_df = train_test_split(
      temp_df,
      test_size=test_fraction,
      random_state=RANDOM_STATE,
      stratify=temp_df["label"],
  )

  return (
      train_df.reset_index(drop=True),
      val_df.reset_index(drop=True),
      test_df.reset_index(drop=True),
  )


def validate_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
  """Ensure no data leakage between splits and verify complete class coverage."""
  # Verify mutually exclusive image IDs across subsets
  train_ids = set(train_df["image_id"])
  val_ids = set(val_df["image_id"])
  test_ids = set(test_df["image_id"])

  if (train_ids & val_ids) or (train_ids & test_ids) or (val_ids & test_ids):
    raise ValueError("Data leakage detected: Overlapping image IDs found!")

  # Ensure all distinct classes are preserved in every subset
  total_classes = len(
      set(train_df["label"]) | set(val_df["label"]) | set(test_df["label"])
  )
  for name, subset in [
      ("Train", train_df),
      ("Validation", val_df),
      ("Test", test_df),
  ]:
    if subset["label"].nunique() != total_classes:
      raise ValueError(f"{name} subset is missing one or more classes!")


def print_and_save_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
  """Print split distribution summaries and persist dataframes to CSV files."""
  total = len(train_df) + len(val_df) + len(test_df)
  print(
      f"\nDataset Split Summary:\n"
      f"- Train      : {len(train_df):4d} samples ({len(train_df)/total:.1%})\n"
      f"- Validation : {len(val_df):4d} samples ({len(val_df)/total:.1%})\n"
      f"- Test       : {len(test_df):4d} samples ({len(test_df)/total:.1%})"
  )

  splits = [
      (TRAIN_SPLIT_PATH, train_df, "Train"),
      (VAL_SPLIT_PATH, val_df, "Validation"),
      (TEST_SPLIT_PATH, test_df, "Test"),
  ]

  print("\nSaved split files:")
  for path, subset_df, name in splits:
    path.parent.mkdir(parents=True, exist_ok=True)
    subset_df.to_csv(path, index=False)
    print(f"- {name:12}: {path}")


def main() -> None:
  labels_df = load_labels()
  train_df, val_df, test_df = split_train_val_test(labels_df)
  validate_splits(train_df, val_df, test_df)
  print_and_save_splits(train_df, val_df, test_df)


if __name__ == "__main__":
  main()
