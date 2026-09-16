# Plant Leaf Identification using Traditional Computer Vision & Machine Learning

This repository implements a complete pipeline for plant leaf species recognition using classical Computer Vision feature descriptors (HOG, LBP) and Support Vector Machines (SVM) on the **Flavia Dataset** (32 plant species, 1,907 images).

---

## 📂 Repository Structure

```text
CV_leaf_identification/
└── leaf_hog_lbp_svm/
    ├── data/
    │   ├── raw/
    │   │   └── Flavia/              # Raw image files (1001.jpg - 3621.jpg) [excluded from Git]
    │   └── splits/                  # Generated train/val/test splits [generated at runtime]
    ├── demo/
    │   ├── leaf_identification_demo.ipynb       # Interactive demo notebook
    │   └── plant_leaf_recognition_pipeline.ipynb # End-to-end pipeline walkthrough
    ├── models/                      # Saved SVM models (*.joblib) [generated at runtime]
    ├── results/                     # Figures, metrics, and prediction tables [generated at runtime]
    │   ├── figures/                 # Confusion matrices, error cases, visualizations
    │   ├── metrics/                 # JSON metrics and evaluation reports
    │   └── predictions/             # Prediction CSVs and demo charts
    ├── src/                         # Modular Python source code
    │   ├── __init__.py
    │   ├── config.py                # Configuration and class mappings
    │   ├── preprocess.py            # Otsu segmentation & moment-based alignment
    │   ├── features.py              # HOG and Uniform LBP feature extraction
    │   ├── data_loader.py           # Dataset loaders and path verification
    │   ├── make_labels.py           # Label extraction from filename ranges
    │   ├── split_data.py            # Stratified Train/Val/Test splitting
    │   ├── train.py                 # Hyperparameter tuning on validation set
    │   ├── compare_models.py        # Validation comparison charts
    │   ├── evaluate.py              # Test set evaluation and confusion matrix
    │   ├── visualize.py             # Feature visualization for individual images
    │   └── demo.py                  # Single-image prediction CLI
    ├── requirements.txt             # Python dependencies
    └── README.md                    # Project-specific documentation
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
Download the **Flavia Leaf Image Dataset** from [Flavia Official Site](https://flavia.sourceforge.net/) and extract all images into `leaf_hog_lbp_svm/data/raw/Flavia/`.

### 3. Pipeline Execution
```powershell
# Generate dataset metadata (32 classes by default, or use --num-classes 10 for baseline)
python -m src.make_labels --num-classes 32
# or: python -m src.make_labels --num-classes 10

# Perform stratified train/val/test splitting (70% / 15% / 15%)
python -m src.split_data

# Train and tune SVM on individual feature representations
python -m src.train --feature-set hog
python -m src.train --feature-set lbp
python -m src.train --feature-set hog_lbp

# Compare validation performance
python -m src.compare_models

# Evaluate the best model on the unseen test set
python -m src.evaluate --feature-set hog_lbp

# Run inference on a test image
python -m src.demo --image data/raw/Flavia/1001.jpg
```
