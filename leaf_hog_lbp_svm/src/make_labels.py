import pandas as pd

from src.config import CLASS_RANGES, LABELS_PATH, RAW_DATA_DIR


def build_labels_dataframe() -> pd.DataFrame:
    """Create a metadata table for 10 Flavia classes."""
    rows = []

    for label, class_name, start_id, end_id in CLASS_RANGES:
        for image_id in range(start_id, end_id + 1):
            image_path = RAW_DATA_DIR / f"{image_id}.jpg"

            if not image_path.exists():
                print(f"Warning: missing image, skipped: {image_path.name}")
                continue

            rows.append(
                {
                    "image_id": image_id,
                    "filename": image_path.name,
                    "image_path": str(image_path),
                    "label": label,
                    "class_name": class_name,
                }
            )

    labels_dataframe = pd.DataFrame(rows)

    if labels_dataframe.empty:
        raise RuntimeError(
            f"No images were found in: {RAW_DATA_DIR}\n"
            "Check that the Flavia images are extracted correctly."
        )

    return labels_dataframe.sort_values(
        by=["label", "image_id"]
    ).reset_index(drop=True)


def validate_labels(labels_dataframe: pd.DataFrame) -> None:
    """Check 10 classes are present and files are unique."""
    expected_labels = {class_info[0] for class_info in CLASS_RANGES}
    actual_labels = set(labels_dataframe["label"].unique())

    missing_labels = expected_labels - actual_labels
    if missing_labels:
        raise ValueError(
            f"Missing labels in the dataset: {sorted(missing_labels)}"
        )

    duplicate_count = labels_dataframe["image_path"].duplicated().sum()
    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate image paths."
        )

    print("\nImages per class:")
    print(
        labels_dataframe.groupby(
            ["label", "class_name"]
        ).size().to_string()
    )

    print(f"\nTotal images: {len(labels_dataframe)}")


def main() -> None:
    labels_dataframe = build_labels_dataframe()
    validate_labels(labels_dataframe)

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels_dataframe.to_csv(LABELS_PATH, index=False)

    print(f"\nSaved labels CSV to: {LABELS_PATH}")


if __name__ == "__main__":
    main()