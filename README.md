# Loan Default Prediction — End-to-End ML Pipeline

Predicting loan default risk using the Lending Club dataset (2007–2018), built as a complete, leakage-checked ML pipeline from raw data to a deployable model.

## Overview

This project predicts whether a borrower will default on a loan, using only information available **at the time of loan origination** — a strict constraint that required catching and removing several subtle data leakage issues along the way (see below).

- **Dataset:** Lending Club accepted loans (~1.3M rows, 151 raw columns)
- **Target:** Binary — `Default` (1) vs `Fully Paid` (0)
- **Final model:** Logistic Regression (`class_weight="balanced"`)
- **Test ROC-AUC:** 0.726
- **Test PR-AUC:** 0.400

## Pipeline

| Stage | Description |
|---|---|
| 1–2 | Load raw data, memory optimization (3.51 GB → 0.94 GB), filter to completed loans, encode binary target |
| 3–5 | Feature audit, classify & remove constant / high-missing / identifier / leakage-prone columns |
| 6–8 | Missing value analysis and imputation |
| 9–10 | Date-based and domain feature engineering (e.g. credit history length) |
| 11 | Time-aware train/test split |
| 12–13 | Preprocessing pipeline (median/mode imputation, scaling, encoding) |
| 14–16 | Baseline (Logistic Regression) and LightGBM model training |
| 17–18 | Feature importance review → **discovered and removed 3 leakage features** |
| 19–20 | Final evaluation, confusion matrix, ROC/PR curves, threshold tuning |
| 21 | Model, preprocessor, and metadata saved as reusable artifacts |

## Key Finding: Data Leakage

An early version of the model scored **0.97 ROC-AUC** — suspiciously high for this problem. Investigating feature importance revealed the top predictors were:

- `last_fico_range_high` / `last_fico_range_low` — FICO scores *re-pulled during the loan's life*, which reflect a borrower's declining credit standing **after** they've already started missing payments.
- `debt_settlement_flag` — only set **after** a borrower enters financial distress.

Both are consequences of default, not predictors of it. After removing them (along with non-predictive identifier/text columns like `url` and `emp_title`), the honest, leakage-free performance dropped to **0.726 ROC-AUC** — a realistic number in line with published benchmarks for this dataset using only origination-time features.

This is the headline result of the project: **catching a leakage bug that would have made the model look great in a demo but fail in production**, since post-origination fields don't exist yet when a real application is scored.

## Model Selection

| Model | Test ROC-AUC | Test PR-AUC |
|---|---|---|
| Logistic Regression | **0.726** | **0.400** |
| LightGBM (default params) | 0.719 | 0.391 |
| LightGBM (regularized) | 0.711 | 0.377 |

LightGBM was tuned with regularization (limited depth, min child samples, subsampling, L1/L2 penalties) but still underperformed and overfit within a handful of boosting rounds. This suggests the true relationship between these features and default risk is largely linear — a legitimate and common outcome in credit risk modeling, where regularized linear models are often preferred in production for this reason, plus their interpretability.

## Results (Final Model, threshold = 0.5)

| | Precision | Recall | F1 |
|---|---|---|---|
| No Default | 0.88 | 0.67 | 0.76 |
| **Default** | 0.34 | 0.66 | 0.45 |

A threshold sweep confirmed 0.5 is already F1-optimal for this model. The model favors **recall over precision** on the default class (catches ~66% of true defaulters), a deliberate trade-off via `class_weight="balanced"`, appropriate for a risk use case where missing a defaulter is typically costlier than a false alarm.

## Artifacts

- `notebooks/loan_default_prediction.ipynb` — full pipeline notebook
- `loan_default_model_logreg.joblib` — trained model
- `preprocessor_final.joblib` — fitted preprocessing pipeline
- `model_metadata.json` — model card (metrics, features removed, notes)
- `final_evaluation_plots.png` — confusion matrix, ROC curve, PR curve
- `feature_audit.csv` — full feature classification audit

## Tech Stack

Python, pandas, scikit-learn, LightGBM, matplotlib

## Notes / Future Work

- Could explore SHAP for per-prediction explainability
- Could add a simple inference script / API endpoint for real applications
- Threshold could be tuned further against a specific business cost matrix (cost of false negative vs false positive) rather than F1
