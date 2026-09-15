# Leaf Classification with HOG, LBP, and SVM

Traditional computer vision pipeline for classifying leaf images from a 10-class subset of the Flavia dataset.

## Features

- HOG feature extraction
- Uniform LBP histogram extraction
- RBF-SVM classification
- Validation comparison: HOG, LBP, and HOG + LBP

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## Setup

Create and activate a virtual environment.

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Dataset

Download the Flavia Leaf Image Dataset from the [official project page](https://flavia.sourceforge.net/).

Extract the image files into:

```text
data/raw/Flavia/
```

Expected structure:

```text
data/raw/Flavia/
├── 1001.jpg
├── 1002.jpg
└── ...
```

## Usage

Run all commands from the repository root.

```powershell
# Generate labels and train/validation/test splits
python -m src.make_labels
python -m src.split_data

# Visualize HOG and LBP features for one image
python -m src.visualize --image data/raw/Flavia/1001.jpg

# Train feature configurations
python -m src.train --feature-set hog
python -m src.train --feature-set lbp
python -m src.train --feature-set hog_lbp

# Compare validation results
python -m src.compare_models

# Refit the selected model and evaluate on the test split
python -m src.evaluate

# Run prediction demo
python -m src.demo --image data/raw/Flavia/1001.jpg
```

## Project Structure

```text
.
├── data/           # Dataset metadata and splits
├── models/         # Trained model artifacts
├── results/        # Metrics, figures, and predictions
├── src/            # Source code
└── requirements.txt
```