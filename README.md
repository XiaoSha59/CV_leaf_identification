# Plant Leaf Identification using HOG, LBP, and SVM

This repository implements an end-to-end Computer Vision and Machine Learning pipeline for plant leaf species identification on the **Flavia Dataset** (32 plant species, 1,907 images).

The method integrates morphological preprocessing, gradient shape analysis via **Histogram of Oriented Gradients (HOG)**, surface texture encoding via **Local Binary Patterns (LBP)**, feature standardization, and an **RBF-Kernel Support Vector Machine (SVM)**.

---

## 📊 Benchmark Results

Evaluated on a held-out test split (15% stratified test set - 287 images) across all 32 plant species:

| Feature Representation | Dimension | Test Accuracy | Macro F1-Score | Correct / Total |
| :--- | :---: | :---: | :---: | :---: |
| **LBP** (Non-rotation-invariant Uniform) | 59 | 93.38% | 0.9348 | 268 / 287 |
| **HOG** (Cell-based Gradient) | 1,728 | 93.73% | 0.9389 | 269 / 287 |
| **HOG + LBP Combined** | **1,787** | **96.17%** | **0.9622** | **276 / 287** |

---

## 📂 Repository Structure

```text
CV_leaf_identification/
└── leaf_hog_lbp_svm/
    ├── data/
    │   ├── raw/
    │   │   └── Flavia/              # Raw images (1001.jpg - 3621.jpg)
    │   └── splits/                  # Train/Val/Test split CSV files
    ├── models/                      # Trained model artifacts (.joblib)
    ├── results/
    │   ├── figures/                 # Confusion matrices and plots
    │   ├── metrics/                 # JSON evaluation reports
    │   └── predictions/             # Prediction CSV outputs
    ├── src/
    │   ├── __init__.py
    │   ├── config.py                # Pipeline parameters and class definitions
    │   ├── data_loader.py           # Dataset loaders and path verification
    │   ├── make_labels.py           # Dataset indexing and label creation
    │   ├── preprocess.py            # Otsu segmentation, vertical alignment & centering
    │   ├── features.py              # HOG (1,728D) and LBP (59D) feature extractors
    │   ├── split_data.py            # Stratified Train/Val/Test (70/15/15) split
    │   ├── train.py                 # Hyperparameter tuning (Grid search on Val set)
    │   └── evaluate.py              # Test set evaluation and confusion matrix generation
    ├── requirements.txt             # Python dependencies
    └── README.md                    # Sub-package documentation
```

---

## ⚙️ Methodology & Pipeline Overview

```
[Raw Image] 
    │
    ▼
[Morphological Preprocessing]
  ├── 1. Otsu Thresholding (Background segmentation)
  ├── 2. Central Moments Calculation
  ├── 3. Principal Axis Vertical Rotation
  ├── 4. Bounding Box Crop (center_leaf)
  └── 5. Standardized Resize (100 × 134 px)
    │
    ▼
[Feature Extraction]
  ├── HOG: 9 orientations, 8×8 cell, 1×1 block -> 1,728 dimensions
  └── LBP: NRI-Uniform (P=8, R=1, 59 bins)    -> 59 dimensions
    │
    ▼
[Feature Fusion] -> 1,787 dimensions
    │
    ▼
[StandardScaler] -> (Zero-mean, unit-variance normalization)
    │
    ▼
[RBF-Kernel SVM] -> 32-Class Probability Prediction
```

---

## 🚀 Quick Start

### 1. Environment Setup

```powershell
# Create and activate virtual environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# Navigate to project folder and install dependencies
cd leaf_hog_lbp_svm
pip install -r requirements.txt
```

### 2. Dataset Preparation

1. Download the **Flavia Leaf Image Dataset** from the [Flavia Official Page](https://flavia.sourceforge.net/).
2. Extract all images (`1001.jpg` to `3621.jpg`) into `leaf_hog_lbp_svm/data/raw/Flavia/`.

### 3. Pipeline Execution

Run all commands from the `leaf_hog_lbp_svm` directory:

```powershell
# Step 1: Generate dataset labels (Full 32 classes)
python -m src.make_labels --num-classes 32

# Step 2: Create stratified 70/15/15 Train/Validation/Test splits
python -m src.split_data

# Step 3: Train and tune hyperparameters for individual and combined features
python -m src.train --feature-set hog
python -m src.train --feature-set lbp
python -m src.train --feature-set hog_lbp

# Step 4: Evaluate the final model on the held-out test set
python -m src.evaluate --feature-set hog_lbp
```

---

## 📦 Requirements

* Python 3.10+
* `numpy`, `scipy`, `pandas`
* `opencv-python`
* `scikit-image`
* `scikit-learn`
* `matplotlib`, `seaborn`
* `joblib`
