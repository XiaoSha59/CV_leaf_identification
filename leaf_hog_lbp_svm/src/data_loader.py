from pathlib import Path
import pandas as pd

from src.config import (
    TEST_SPLIT_PATH,
    TRAIN_SPLIT_PATH,
    VAL_SPLIT_PATH,
)

SPLIT_PATHS = {
    "train": TRAIN_SPLIT_PATH,
    "val": VAL_SPLIT_PATH,
    "test": TEST_SPLIT_PATH,
}

def load_split(split_name: str) -> pd.DataFrame:
    """Load a saved dataset split by name."""
    if split_name not in SPLIT_PATHS:
        valid_names = ", ".join(SPLIT_PATHS)
        raise ValueError(
            f"Unknown split name: {split_name}. "
            f"Expected one of: {valid_names}"
        )

    split_path = SPLIT_PATHS[split_name]

    if not split_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}\n"
            "Run `python -m src.split_data` first."
        )

    dataframe = pd.read_csv(split_path)

    required_columns = {
        "image_id",
        "filename",
        "image_path",
        "label",
        "class_name",
    }

    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"{split_path.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        raise ValueError(f"{split_path.name} is empty.")

    return dataframe


def get_class_names(dataframe: pd.DataFrame) -> list[str]:
    """Return class names ordered by numeric label."""
    class_table = (
        dataframe[["label", "class_name"]]
        .drop_duplicates()
        .sort_values("label")
    )

    return class_table["class_name"].tolist()


def verify_image_paths(dataframe: pd.DataFrame) -> None:
    """Fail early if any image path in a split does not exist."""
    missing_paths = [
        image_path
        for image_path in dataframe["image_path"]
        if not Path(image_path).exists()
    ]

    if missing_paths:
        preview = "\n".join(missing_paths[:5])
        raise FileNotFoundError(
            f"Found {len(missing_paths)} missing image files. "
            f"First paths:\n{preview}"
        )