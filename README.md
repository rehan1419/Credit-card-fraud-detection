# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions from highly imbalanced transaction data, comparing Logistic Regression and tuned Random Forest models with full EDA, class-imbalance handling, and precision/recall-focused evaluation.

## Overview

Credit card fraud is a classic imbalanced classification problem — fraudulent transactions make up a tiny fraction of all transactions, so accuracy alone is a misleading metric. This project walks through the full pipeline: exploratory data analysis, feature scaling, baseline modeling, a tuned ensemble model, and evaluation using metrics suited to imbalanced data (ROC-AUC, precision/recall, PR-AUC, confusion matrices).

## Dataset

Uses the well-known **`creditcard.csv`** dataset (anonymized, PCA-transformed European cardholder transactions from September 2013):

- **284,807** total transactions
- **492** fraudulent transactions (**0.17%** of the data) — severely imbalanced
- 30 features: `Time`, `Amount`, and 28 PCA-anonymized features (`V1`–`V28`)
- Target: `Class` (0 = normal, 1 = fraud)

> Note: `creditcard.csv` is not included in this repo (likely due to file size / licensing on the original Kaggle dataset). You'll need to download it separately — see [Setup](#setup) below.

## Approach

1. **EDA** — inspect dataset shape, check for missing values, examine class distribution, and visualize transaction amount patterns for fraud vs. normal transactions.
2. **Feature engineering** — scale `Amount` and `Time` using `StandardScaler` (the `V1`–`V28` features are already PCA-transformed and scaled); drop the original unscaled columns.
3. **Train/test split** — 80/20 split, stratified on the `Class` label to preserve the fraud ratio in both sets.
4. **Baseline model** — Logistic Regression with `class_weight="balanced"` to counteract class imbalance.
5. **Random Forest model** — 200 trees, max depth 10, also using `class_weight="balanced"`.
6. **Hyperparameter tuning** — `GridSearchCV` over `n_estimators` (100/200/300) and `max_depth` (5/10/15), optimizing for F1 score with 3-fold cross-validation.
7. **Evaluation** — classification reports, ROC-AUC, confusion matrices, and a precision-recall curve (more informative than ROC-AUC alone on highly imbalanced data).
8. **Feature importance** — identifies which PCA components the tuned Random Forest relies on most.

## Results

| Model | ROC-AUC | Fraud Precision | Fraud Recall | Fraud F1 |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.972 | 0.06 | 0.92 | 0.11 |
| Random Forest (default params) | 0.986 | 0.81 | 0.81 | 0.81 |
| **Random Forest (tuned)** | 0.965 | **0.90** | 0.80 | **0.84** |

**Best tuned parameters:** `max_depth=15`, `n_estimators=100`

**Top predictive features:** `V14`, `V10`, `V4`, `V17`, `V11`, `V12`, `V3`, `V16`, `V7`, `V2`

The Logistic Regression baseline achieves very high recall (catches 92% of fraud) but at the cost of extremely poor precision (94% of its fraud flags are false alarms) — a direct consequence of class-weighting on such a skewed dataset. The tuned Random Forest offers a far more usable balance, catching 80% of fraud while being right 90% of the time it flags a transaction.

## Repository Structure

```
Credit-card-fraud-detection/
└── fraud_detector.ipynb    # Full pipeline: EDA, preprocessing, modeling, tuning, evaluation
```

## Tech Stack

- **pandas / numpy** — data loading and manipulation
- **matplotlib** — EDA and evaluation visualizations (class distribution, amount histograms, confusion matrix, PR curve, feature importance)
- **scikit-learn** — `StandardScaler`, `LogisticRegression`, `RandomForestClassifier`, `GridSearchCV`, and imbalanced-classification metrics (`roc_auc_score`, `precision_recall_curve`, `classification_report`)

## Setup

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

```bash
git clone https://github.com/rehan1419/Credit-card-fraud-detection.git
cd Credit-card-fraud-detection
pip install pandas numpy matplotlib scikit-learn jupyter
```

### Get the dataset

Download `creditcard.csv` from the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root (same folder as `fraud_detector.ipynb`).

### Running

```bash
jupyter notebook fraud_detector.ipynb
```

Run all cells in order. The notebook will print dataset stats, render EDA plots, train both models, run the grid search, and print the final comparison table.

## Future Improvements

- Try SMOTE or other resampling techniques and compare against the `class_weight="balanced"` approach used here
- Add XGBoost/LightGBM as an additional model for comparison
- Tune the classification threshold explicitly (rather than the default 0.5) using the precision-recall curve to hit a target precision/recall trade-off
- Wrap the final model in a simple script or API for scoring new transactions
- Add cross-validation on the final tuned model for a more robust performance estimate
