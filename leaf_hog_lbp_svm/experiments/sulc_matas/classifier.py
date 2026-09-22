import sys
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
from sklearn.kernel_approximation import AdditiveChi2Sampler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from sklearn.multiclass import OneVsRestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class FfirstRegionClassifier:
    def __init__(self, C: float = 1000.0):
        self.C = C
        self.chi2_sampler = AdditiveChi2Sampler(sample_steps=2)
        self.base_clf = OneVsRestClassifier(
            LogisticRegression(
                C=C,
                max_iter=5000,
                solver="lbfgs",
                random_state=42,
            )
        )
        self.pipeline = Pipeline(
            steps=[
                ("chi2", self.chi2_sampler),
                ("clf", self.base_clf),
            ]
        )

    def fit(self, X_multiscale_list: List[List[np.ndarray]], y_labels: np.ndarray):
        X_expanded = []
        y_expanded = []
        
        for img_descriptors, label in zip(X_multiscale_list, y_labels):
            for desc in img_descriptors:
                X_expanded.append(desc)
                y_expanded.append(label)
                
        X_train = np.vstack(X_expanded)
        y_train = np.array(y_expanded, dtype=np.int64)
        
        self.pipeline.fit(X_train, y_train)
        return self

    def predict_proba_image(self, img_descriptors: List[np.ndarray]) -> np.ndarray:
        X_img = np.vstack(img_descriptors)
        # Probabilities for all 3 scales: shape (3, n_classes)
        probs_scales = self.pipeline.predict_proba(X_img)
        # Max posterior probability over all scales (Section 3.3)
        max_probs = np.max(probs_scales, axis=0)
        return max_probs

    def predict_proba_dataset(self, X_multiscale_list: List[List[np.ndarray]]) -> np.ndarray:
        all_probs = []
        for img_descriptors in X_multiscale_list:
            probs = self.predict_proba_image(img_descriptors)
            all_probs.append(probs)
        return np.vstack(all_probs)

    def predict(self, X_multiscale_list: List[List[np.ndarray]]) -> np.ndarray:
        probs = self.predict_proba_dataset(X_multiscale_list)
        return np.argmax(probs, axis=1)


def fuse_probabilities(
    probs_interior: np.ndarray,
    probs_border: np.ndarray,
    fusion_method: str = "product",
) -> Tuple[np.ndarray, np.ndarray]:
    if fusion_method == "product":
        fused_probs = probs_interior * probs_border
    elif fusion_method == "sum":
        fused_probs = probs_interior + probs_border
    else:
        raise ValueError("fusion_method must be either 'product' or 'sum'")
        
    predictions = np.argmax(fused_probs, axis=1)
    return fused_probs, predictions


if __name__ == "__main__":
    dummy_train = [[np.random.rand(456).astype(np.float32) for _ in range(3)] for _ in range(20)]
    dummy_labels = np.random.randint(0, 5, size=20)
    
    clf = FfirstRegionClassifier(C=100.0)
    clf.fit(dummy_train, dummy_labels)
    
    dummy_test = [[np.random.rand(456).astype(np.float32) for _ in range(3)] for _ in range(5)]
    probs = clf.predict_proba_dataset(dummy_test)
    preds = clf.predict(dummy_test)
    
    print(f"Predicted probabilities shape: {probs.shape} (Expected 5 samples x 5 classes)")
    print(f"Predicted class labels: {preds}")
