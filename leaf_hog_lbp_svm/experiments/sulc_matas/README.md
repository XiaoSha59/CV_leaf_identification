# Reproduction: Texture-Based Leaf Identification (Sulc & Matas, ECCV 2014)

This directory contains the formal reproduction of the state-of-the-art leaf classification paper:
> **Milan Sulc and Jiri Matas**, *"Texture-Based Leaf Identification"*, European Conference on Computer Vision (ECCV) Workshops, 2014.

The method is evaluated on the **Flavia Dataset** (32 species, 1,907 images) under a stratified 70/15/15 Train/Val/Test split.

---

## 🏗️ Method & Pipeline Architecture

```text
                     [Input Leaf Image (RGB / Grayscale)]
                                      │
                                      ▼
                        [1. Otsu Leaf Segmentation]
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        [Interior Mask: M_int]                [Border Mask: M_border]
        (Erosion r=35px: Venation)            (Margin & Leaf Teeth)
                   │                                     │
                   ▼                                     ▼
      [2. 8-Scale Gaussian Space]           [2. 8-Scale Gaussian Space]
      ├── Radii R: 1 ... 11.314 px          ├── Radii R: 1 ... 11.314 px
      └── Sigma = R / 2                     └── Sigma = R / 2
                   │                                     │
                   ▼                                     ▼
      [3. Multi-Scale LBP-HF-S-M]           [3. Multi-Scale LBP-HF-S-M]
      ├── CLBP-Sign (38D)                   ├── CLBP-Sign (38D)
      ├── CLBP-Mag  (38D)                   ├── CLBP-Mag  (38D)
      ├── 1D Discrete Fourier (76D/scale)   ├── 1D Discrete Fourier (76D/scale)
      └── 6 Scales x 76D = 456D             └── 6 Scales x 76D = 456D
          (3 Channels: 3 x 456D)                (3 Channels: 3 x 456D)
                   │                                     │
                   ▼                                     ▼
      [4. Additive Chi-Square Map]          [4. Additive Chi-Square Map]
                   │                                     │
                   ▼                                     ▼
      [5. Linear SVM + Platt Scaling]       [5. Linear SVM + Platt Scaling]
                   │                                     │
                   ▼                                     ▼
          P(Class | Interior)                   P(Class | Border)
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                           [6. Late Decision Fusion]
                                      ├── Sum Rule: P(int) + P(border)      --> 97.21%
                                      └── Product Rule: P(int) x P(border)  --> 97.56%
                                      │
                                      ▼
                       [Final 32-Species Classification]
```

---

## Method Overview

The Sulc & Matas pipeline models leaf recognition through multi-scale rotation-invariant texture descriptors extracted from separate anatomical regions of the leaf:

1. **Leaf Segmentation & Region Decomposition (`segmentation.py`)**:
   - Otsu thresholding with convex hull / hole-filling.
   - Morphological erosion with a circular structuring element (radius = 35 px) to decompose the leaf into:
     - **Interior region (M_int)**: Lamina venation and surface micro-texture.
     - **Border region (M_border)**: Margin serrations, teeth, and leaf shape contour.

2. **Completed LBP Histogram Fourier Descriptors (`lbp_hf.py`)**:
   - Computes both **CLBP-Sign** (difference signs) and **CLBP-Magnitude** (gradient intensity contrast relative to mean gradient).
   - Applies 1D Discrete Fourier Transform (DFT) over rotation-shifted uniform LBP bin groups to achieve continuous rotation invariance (76D descriptor per scale).

3. **Multi-Scale Gaussian Filtering (`multiscale.py`)**:
   - 8-scale space representation with radii R in {1, 1.414, 2, 2.828, 4, 5.657, 8, 11.314} and matching Gaussian smoothing (sigma = R / 2).
   - Yields a 456D feature vector (6 x 76D) per leaf region for each color channel.

4. **Classifier & Late Fusion (`classifier.py`)**:
   - Homogeneous Additive Chi-Square Feature Map (`AdditiveChi2Sampler`) transforming non-linear Chi-square kernel space to linear space.
   - One-vs-Rest Linear SVM with Platt scaling probability estimation.
   - Late decision fusion combining posterior probabilities from Interior and Border classifiers using **Product Rule** (Pi) and **Sum Rule** (Sigma).

---

## File Structure

```text
leaf_hog_lbp_svm/experiments/sulc_matas/
├── segmentation.py           # Otsu thresholding & interior/border morphological separation
├── lbp_hf.py                 # Completed LBP (CLBP-S, CLBP-M) & 1D DFT rotation invariance
├── multiscale.py             # 8-scale space filtering & multi-scale descriptor concatenation
├── classifier.py             # Additive Chi-Square map + Linear SVM + Platt Late Fusion
├── run_reproduction_flavia.py# Parallel multi-core benchmark runner
└── README.md                 # Experiment documentation
```

---

## How to Run

Run the reproduction script from the `leaf_hog_lbp_svm` root directory:

```powershell
# Navigate to leaf_hog_lbp_svm directory
cd leaf_hog_lbp_svm

# Execute the parallel multi-threaded runner (utilizes all CPU cores)
python -m experiments.sulc_matas.run_reproduction_flavia
```

---

## Experimental Results on Flavia Test Set (287 images, 32 classes)

| Model / Feature Configuration | Region | Fusion Rule | Feature Dim | Test Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| Baseline HOG 2x2 + PCA (k=64) + LBP (59D) | Whole Leaf | Early Concatenation | 123D | 97.56% (280/287) |
| Sulc & Matas - Ffirst (Interior only) | Interior | None | 3 x 456D | 95.82% (275/287) |
| Sulc & Matas - Ffirst (Border only) | Border | None | 3 x 456D | 94.77% (272/287) |
| **Sulc & Matas - Ffirst (Interior + Border)** | Both | **Sum Rule** | 2 x (3 x 456D) | **97.21% (279/287)** |
| **Sulc & Matas - Ffirst (Interior + Border)** | Both | **Product Rule** | 2 x (3 x 456D) | **97.56% (280/287)** |
