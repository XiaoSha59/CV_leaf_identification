# Leaf Classification with HOG, LBP, and SVM

This package implements an end-to-end Computer Vision pipeline for plant leaf species recognition on the **Flavia Dataset** (32 classes, 1,907 images).

---

## 📊 Performance Summary

| Feature Configuration | Dimensions | Test Accuracy | Macro F1 |
| :--- | :---: | :---: | :---: |
| **LBP** (NRI-Uniform) | 59 | 93.38% | 0.9348 |
| **HOG** (Cell-based) | 1,728 | 93.73% | 0.9389 |
| **HOG + LBP Combined** | **1,787** | **96.17%** | **0.9622** |

---

## 🚀 Execution Workflow

Run all commands from the `leaf_hog_lbp_svm` directory:

```powershell
# 1. Generate dataset metadata (32 classes)
python -m src.make_labels --num-classes 32

# 2. Generate stratified splits (70% train, 15% val, 15% test)
python -m src.split_data

# 3. Train models with hyperparameter validation
python -m src.train --feature-set hog
python -m src.train --feature-set lbp
python -m src.train --feature-set hog_lbp

# 4. Evaluate final refit model on test set
python -m src.evaluate --feature-set hog_lbp
```

---

## 📁 Source Modules

* `src/config.py`: Central parameters (image size 100x134, HOG cell 8x8, LBP 59-bins, class definitions).
* `src/preprocess.py`: Otsu thresholding, principal moment alignment, centroid cropping, and normalization.
* `src/features.py`: HOG (1,728D) and Non-rotation-invariant Uniform LBP (59D) extractors.
* `src/data_loader.py`: Dataset batching and verification routines.
* `src/make_labels.py`: Range-based dataset labeling.
* `src/split_data.py`: Stratified Train/Val/Test partitioning.
* `src/train.py`: Grid search tuning on validation data.
* `src/evaluate.py`: Final model refitting and test evaluation with confusion matrix.