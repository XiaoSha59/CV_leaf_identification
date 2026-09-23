from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, PCA_HOG_COMPONENTS, RANDOM_STATE


def fit_pca(
    X_features: np.ndarray,
    n_components: int | float = PCA_HOG_COMPONENTS,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Fit StandardScaler + PCA pipeline on features."""
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=n_components,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipeline.fit(X_features)

    pca_step = pipeline.named_steps["pca"]
    var_retained = float(np.sum(pca_step.explained_variance_ratio_) * 100.0)
    print(
        f"Fitted Scaled PCA: {X_features.shape[1]}D -> {pca_step.n_components_}D "
        f"({var_retained:.2f}% cumulative variance retained)"
    )
    return pipeline


def transform_pca(
    X_features: np.ndarray,
    pca_pipeline: Pipeline | PCA,
) -> np.ndarray:
    """Project features using fitted PCA pipeline."""
    return pca_pipeline.transform(X_features).astype(np.float32)


def fit_transform_pca(
    X_features: np.ndarray,
    n_components: int | float = PCA_HOG_COMPONENTS,
    random_state: int = RANDOM_STATE,
) -> tuple[Pipeline, np.ndarray]:
    """Fit PCA and transform features."""
    pipeline = fit_pca(
        X_features,
        n_components=n_components,
        random_state=random_state,
    )
    X_reduced = transform_pca(X_features, pipeline)
    return pipeline, X_reduced


def save_pca(
    pca_pipeline: Pipeline | PCA,
    output_path: str | Path | None = None,
) -> Path:
    """Save fitted PCA pipeline to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if isinstance(pca_pipeline, Pipeline):
        n_comp = pca_pipeline.named_steps["pca"].n_components_
    else:
        n_comp = pca_pipeline.n_components_

    if output_path is None:
        output_path = MODELS_DIR / f"hog_scaled_pca_{n_comp}.joblib"
    else:
        output_path = Path(output_path)

    joblib.dump(pca_pipeline, output_path)
    print(f"Saved Scaled PCA pipeline to: {output_path}")
    return output_path


def load_pca(model_path: str | Path) -> Pipeline | PCA:
    """Load saved PCA pipeline from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"PCA model file not found: {path}")
    return joblib.load(path)
