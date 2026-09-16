import argparse
import pandas as pd

from src.config import CLASS_RANGES, LABELS_PATH, RAW_DATA_DIR


def build_labels_dataframe(num_classes: int = 32) -> pd.DataFrame:
    """Create a metadata table for Flavia dataset (supports 10-class subset or full 32 classes)."""
    selected_ranges = CLASS_RANGES[:num_classes]
    rows = []

    for label, class_name, start_id, end_id in selected_ranges:
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


def validate_labels(labels_dataframe: pd.DataFrame, num_classes: int = 32) -> None:
    """Check expected classes are present and file paths are unique."""
    expected_labels = {class_info[0] for class_info in CLASS_RANGES[:num_classes]}
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

    print(f"\nTotal classes: {labels_dataframe['label'].nunique()}")
    print(f"Total images: {len(labels_dataframe)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate labels metadata for Flavia dataset (10-class baseline or full 32 classes)."
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=32,
        choices=[10, 32],
        help="Number of classes to include: 10 (initial baseline subset) or 32 (full dataset, default).",
    )
    args = parser.parse_args()

    print(f"Building labels dataframe for {args.num_classes} classes...")
    labels_dataframe = build_labels_dataframe(num_classes=args.num_classes)
    validate_labels(labels_dataframe, num_classes=args.num_classes)

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    labels_dataframe.to_csv(LABELS_PATH, index=False)

    print(f"\nSaved labels CSV to: {LABELS_PATH}")


if __name__ == "__main__":
    main()