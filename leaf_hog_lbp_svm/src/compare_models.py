import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR, METRICS_DIR


FEATURE_SETS = ["hog", "lbp", "hog_lbp"]

MODEL_LABELS = {
    "hog": "HOG + RBF-SVM",
    "lbp": "Uniform LBP + RBF-SVM",
    "hog_lbp": "HOG + Uniform LBP + RBF-SVM",
}


def load_validation_results() -> pd.DataFrame:
    """Load validation metrics saved by train.py for all feature sets."""
    rows = []

    for feature_set in FEATURE_SETS:
        metrics_path = METRICS_DIR / f"{feature_set}_validation_metrics.json"

        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Missing metrics file: {metrics_path}\n"
                f"Run `python -m src.train --feature-set {feature_set}` first."
            )

        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

        best_result = metrics["best_validation_result"]

        rows.append(
            {
                "feature_set": feature_set,
                "model": MODEL_LABELS[feature_set],
                "feature_dimension": metrics["feature_dimension"],
                "C": best_result["C"],
                "gamma": best_result["gamma"],
                "validation_accuracy": best_result["val_accuracy"],
                "validation_macro_f1": best_result["val_macro_f1"],
                "train_seconds": best_result["train_seconds"],
            }
        )

    return pd.DataFrame(rows)


def save_metrics_table(results_dataframe: pd.DataFrame) -> Path:
    """Save a concise CSV table of model comparisons."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = METRICS_DIR / "validation_model_comparison.csv"
    results_dataframe.to_csv(
        output_path,
        index=False,
        float_format="%.6f",
    )
    return output_path


def annotate_bars(axis: plt.Axes) -> None:
    """Write the score value above each bar."""
    for container in axis.containers:
        axis.bar_label(
            container,
            fmt="%.3f",
            padding=3,
            fontsize=10,
        )


def plot_model_comparison(results_dataframe: pd.DataFrame) -> Path:
    """Create validation Accuracy and Macro-F1 comparison charts."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_dataframe = results_dataframe.copy()
    plot_dataframe["validation_accuracy"] *= 100
    plot_dataframe["validation_macro_f1"] *= 100

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        sharey=True,
    )

    colors = ["#4C78A8", "#59A14F", "#F28E2B"]

    accuracy_axis = axes[0]
    accuracy_axis.bar(
        plot_dataframe["model"],
        plot_dataframe["validation_accuracy"],
        color=colors,
        width=0.55,
    )
    accuracy_axis.set_title("Validation Accuracy", fontsize=12, fontweight="bold")
    accuracy_axis.set_ylabel("Score (%)", fontsize=11)
    accuracy_axis.set_ylim(0, 100)
    accuracy_axis.tick_params(axis="x", rotation=15)
    accuracy_axis.grid(axis="y", alpha=0.25)
    annotate_bars(accuracy_axis)

    f1_axis = axes[1]
    f1_axis.bar(
        plot_dataframe["model"],
        plot_dataframe["validation_macro_f1"],
        color=colors,
        width=0.55,
    )
    f1_axis.set_title("Validation Macro F1-score", fontsize=12, fontweight="bold")
    f1_axis.set_ylabel("Score (%)", fontsize=11)
    f1_axis.set_ylim(0, 100)
    f1_axis.tick_params(axis="x", rotation=15)
    f1_axis.grid(axis="y", alpha=0.25)
    annotate_bars(f1_axis)

    best_idx = results_dataframe["validation_macro_f1"].idxmax()
    best_model_name = results_dataframe.loc[best_idx, "model"]
    best_f1_val = results_dataframe.loc[best_idx, "validation_macro_f1"] * 100.0

    fig.suptitle(
        "Validation Performance Comparison of Handcrafted Features\n"
        f"Model selection metric: Validation Macro F1 | Selected: {best_model_name} ({best_f1_val:.2f}%)",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()

    output_path = FIGURES_DIR / "model_comparison_validation.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare validation performance across HOG, LBP, and HOG+LBP."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (optional).",
    )
    _ = parser.parse_args()

    results_dataframe = load_validation_results()

    print("\nValidation model comparison:")
    print(
        results_dataframe[
            [
                "model",
                "feature_dimension",
                "C",
                "gamma",
                "validation_accuracy",
                "validation_macro_f1",
            ]
        ].to_string(index=False)
    )

    table_path = save_metrics_table(results_dataframe)
    figure_path = plot_model_comparison(results_dataframe)

    best_row = results_dataframe.loc[
        results_dataframe["validation_macro_f1"].idxmax()
    ]

    print(f"\nSaved comparison table: {table_path}")
    print(f"Saved comparison figure: {figure_path}")
    print(
        f"\nBest validation model: {best_row['model']} "
        f"(Macro F1 = {best_row['validation_macro_f1']:.4f})"
    )


if __name__ == "__main__":
    main()