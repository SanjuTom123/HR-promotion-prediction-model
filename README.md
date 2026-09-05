"# HR-promotion-prediction-model" 

A machine learning pipeline that predicts [what exactly — e.g. employee promotion eligibility] 
using LightGBM with stratified k-fold cross-validation.

## Overview
This script:
- Loads and preprocesses input data from CSV files
- Encodes categorical features using `LabelEncoder`
- Trains a LightGBM classification model with `StratifiedKFold` cross-validation
- Evaluates performance using F1 score
- Outputs predictions to a CSV file

## Requirements
```bash
pip install pandas numpy scikit-learn lightgbm
```

## Files
- `script.py` — main script
- `[file1.csv]` — [what this data is]
- `[file2.csv]` — [what this data is]
- `output.csv` — generated predictions (output)

## How to run
```bash
python script.py
```

## Output
The script generates `[output.csv]` containing [describe columns — e.g. predicted labels/probabilities].
