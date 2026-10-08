# Manufacturing Failure Prediction Dashboard — Design Spec

**Date**: 2026-10-08
**Purpose**: Portfolio project for a Junior Data Scientist application at Michelin. Time budget: ~7-8 hours. Priority: polished, interview-defensible MVP over completeness or production-readiness.

## Problem framing

- **Dataset**: AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository, 10,000 rows, 1 row = 1 machine operating cycle). Downloadable directly, no account needed: `https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip`.
- **Target**: `Machine failure` — binary (0/1), ~3.4% positive rate (imbalanced).
- **Dropped columns** (leakage / no signal):
  - `UDI`, `Product ID` — identifiers only.
  - `TWF`, `HDF`, `PWF`, `OSF`, `RNF` — failure-mode sub-flags. These encode *why* a failure occurred and are only known after the fact, so training on them would leak the target.
- **Features kept**: `Type` (L/M/H quality variant), `Air temperature [K]`, `Process temperature [K]`, `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`.
- **ML problem**: binary classification under class imbalance.
- **Primary metric**: Recall on the failure class (missing a real failure is costlier than a false alarm), reported alongside precision/F1/ROC-AUC. Final decision threshold is chosen from the precision-recall curve to favor recall without collapsing precision, and the tradeoff is stated explicitly rather than hidden behind one number.

## Project structure

```
manufacturing-failure-prediction/
├── data/
│   ├── raw/ai4i2020.csv
│   └── processed/
├── models/
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── evaluation.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Data pipeline (`data_preprocessing.py`)

- Load raw CSV, rename columns to snake_case.
- Drop identifier + leakage columns.
- Defensive checks for missing values / duplicates.
- Feature engineering (two physics-grounded features):
  - `temp_diff_k` = `process_temp_k` − `air_temp_k`
  - `power_w` ≈ `torque_nm` × `rotational_speed_rpm` × (2π/60)
- One-hot encode `type` (L/M/H).
- Stratified 80/20 train/test split (stratified because the positive class is only ~3.4% of rows), saved to `data/processed/`.

## Modeling (`train.py`)

- Baseline: `LogisticRegression(class_weight="balanced")`.
- Comparison: `RandomForestClassifier(class_weight="balanced")`.
- `class_weight="balanced"` handles imbalance without adding SMOTE; SMOTE is named as a future-improvement line in the README.
- Model selection via 5-fold stratified cross-validation on the training split (recall/F1/ROC-AUC), small hyperparameter grid on the Random Forest (`n_estimators`, `max_depth`).
- Held-out test set touched exactly once, for the numbers reported in the dashboard/README.
- Persist `models/model.joblib`, the fitted preprocessing/column spec, and `models/metrics.json` (final test metrics + confusion matrix) so the Streamlit app loads artifacts without retraining.

## Evaluation (`evaluation.py`)

Shared helpers (used by `train.py` and `app.py`): accuracy, precision, recall, F1, ROC-AUC, confusion matrix, precision-recall curve (used to justify the chosen probability threshold).

## Explainability

- Primary: SHAP `TreeExplainer` on the Random Forest — global feature importance (Business Insights page) and per-prediction local explanation (Failure Prediction page).
- Fallback (if SHAP integration stalls past ~20 minutes): `feature_importances_` for global importance; a simple deviation-from-training-mean heuristic for local "top factors."

## Streamlit app (`app.py`)

Sidebar navigation, 5 pages:
1. **Overview** — dataset size, failure rate, model performance headline.
2. **Data Analysis** — failure distribution, feature distributions, correlations.
3. **Model Performance** — confusion matrix, precision/recall/F1/ROC-AUC.
4. **Failure Prediction** — inputs for the 6 real features (ranges from training data), Predict button, failure probability, LOW/MEDIUM/HIGH risk badge (thresholds from the precision-recall curve, documented), SHAP-based top factors for that input.
5. **Business Insights** — patterns the model found, what it can/can't tell us, preventive-maintenance use case.

All pages load persisted artifacts from `models/` — no retraining inside the app.

## README

Sections: Problem, Business Context, Dataset, Approach, Data Preparation, Exploratory Analysis, Model Selection, Model Performance, Explainability, Demo, Limitations, Future Improvements, How to Run. States explicitly this is a portfolio MVP, not production-ready. Limitations to name: synthetic-but-physics-grounded dataset, class imbalance, FP/FN cost asymmetry, no real-time streaming data, generalization caveats.

## Explicit non-goals

- No LangChain/LangGraph, no Kubernetes, no cloud services, no auth, no database, no custom frontend framework.
- No more than 2 models compared (Logistic Regression, Random Forest).
- No SMOTE/oversampling in the MVP (named as future work instead).
- No retraining inside the Streamlit app.
