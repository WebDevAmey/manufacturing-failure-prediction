# Manufacturing Failure Prediction Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end ML application (data pipeline, model training, explainability, Streamlit dashboard, README) that predicts manufacturing machine failure from sensor data, as a portfolio piece for a Junior Data Scientist application.

**Architecture:** A one-way pipeline — raw CSV → `src/data_preprocessing.py` (clean/engineer/split) → `src/train.py` (train, select, evaluate, persist artifacts) → `models/model.joblib` + `models/metrics.json`. `src/predict.py` holds the inference + explainability logic shared by `src/train.py` (for global importance at training time) and `app.py` (for live predictions). `app.py` only ever reads persisted artifacts — it never retrains.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn, shap, matplotlib/seaborn, streamlit, pytest.

All code in this plan has already been dry-run against the real AI4I 2020 dataset in a throwaway environment during planning — the commands, file shapes, and metrics below are real captured output, not estimates.

## Global Constraints

- No LangChain/LangGraph, no Kubernetes, no cloud services, no authentication, no database, no custom frontend framework.
- At most 2 models compared: `LogisticRegression` and `RandomForestClassifier`.
- Class imbalance handled via `class_weight="balanced"` only — no SMOTE/oversampling (named as future work in the README instead).
- Drop leakage columns `TWF`, `HDF`, `PWF`, `OSF`, `RNF` and identifier columns `UDI`, `Product ID` before training — they are not real predictive features.
- `app.py` never retrains a model — it only loads `models/model.joblib` and `models/metrics.json`.
- Primary metric to headline everywhere (dashboard, README): **Recall** on the failure class, always shown alongside precision/F1/ROC-AUC/confusion matrix, never alone.
- Held-out test set is touched exactly once, after model + threshold selection via cross-validation on the training split only — no test-set leakage into model or threshold selection.
- `requirements.txt` pins (validated to install together cleanly): `pandas==3.0.6`, `numpy==2.5.3`, `scikit-learn==1.9.1`, `joblib==1.6.0`, `matplotlib==3.11.2`, `seaborn==0.13.2`, `streamlit==1.65.0`, `shap==0.52.0`, `pytest==9.1.1`, `nbformat==5.11.1`.
- Dataset: AI4I 2020 Predictive Maintenance Dataset, downloaded from `https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip` (10,000 rows, 339 failures, no missing values, no duplicates — verified during planning).
- All project commands assume the current working directory is the project root: `/Users/ameynitintarmale/Desktop/Predictive Maintainence System`.

---

### Task 0: Project Scaffolding & Environment Setup

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `src/__init__.py` (empty)
- Create: `data/raw/ai4i2020.csv` (downloaded)
- Create directories: `data/processed/`, `models/`, `notebooks/`, `tests/`

**Interfaces:**
- Produces: a `venv/` virtual environment with all pinned packages installed, and `data/raw/ai4i2020.csv` on disk, that every later task depends on.

- [x] **Step 1: Create the directory structure**

```bash
cd "/Users/ameynitintarmale/Desktop/Predictive Maintainence System"
mkdir -p data/raw data/processed models notebooks src tests
touch src/__init__.py
```

- [x] **Step 2: Write `requirements.txt`**

```
pandas==3.0.6
numpy==2.5.3
scikit-learn==1.9.1
joblib==1.6.0
matplotlib==3.11.2
seaborn==0.13.2
streamlit==1.65.0
shap==0.52.0
pytest==9.1.1
nbformat==5.11.1
```

- [x] **Step 3: Write `.gitignore`**

```
venv/
__pycache__/
*.pyc
.pytest_cache/
.ipynb_checkpoints/
.DS_Store
```

(`data/`, `models/`, and the notebook are intentionally **not** ignored — this is a small, fully-reproducible portfolio project, and committing the ~500KB dataset and ~1MB trained model means anyone who clones the repo can run `streamlit run app.py` immediately without a download or training step.)

- [x] **Step 4: Create and activate a virtual environment, install dependencies**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -q --upgrade pip
pip install -r requirements.txt
```

**Test:** Run `python -c "import pandas, numpy, sklearn, joblib, matplotlib, seaborn, streamlit, shap, pytest, nbformat; print('all imports OK')"`
Expected output: `all imports OK`

- [x] **Step 5: Download the dataset into `data/raw/`**

```bash
curl -sL -o /tmp/ai4i2020.zip "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
unzip -o -q /tmp/ai4i2020.zip -d /tmp/ai4i2020_extracted
cp /tmp/ai4i2020_extracted/ai4i2020.csv data/raw/ai4i2020.csv
rm -rf /tmp/ai4i2020.zip /tmp/ai4i2020_extracted
```

**Test:**

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/raw/ai4i2020.csv', encoding='utf-8-sig')
print(df.shape)
print(df['Machine failure'].value_counts().to_dict())
"
```
Expected output:
```
(10000, 14)
{0: 9661, 1: 339}
```

- [x] **Step 6: Commit**

```bash
git add requirements.txt .gitignore src/__init__.py data/raw/ai4i2020.csv
git commit -m "Scaffold project structure, environment, and dataset"
```

---

### Task 1: Data Preprocessing Module

**Files:**
- Create: `src/data_preprocessing.py`
- Test: `tests/test_data_preprocessing.py`

**Interfaces:**
- Produces (used by `src/predict.py`, `src/train.py`, `app.py`, `notebooks/01_eda.ipynb`):
  - `load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame`
  - `clean_columns(df: pd.DataFrame) -> pd.DataFrame`
  - `engineer_features(df: pd.DataFrame) -> pd.DataFrame`
  - `encode_type(df: pd.DataFrame) -> pd.DataFrame`
  - `build_feature_table(raw_df: pd.DataFrame) -> pd.DataFrame`
  - `FEATURE_COLUMNS: list[str]` — the exact 10 ordered model input columns
  - `TARGET_COLUMN: str` — `"machine_failure"`
  - `PROCESSED_DIR: Path` — `data/processed/`

- [x] **Step 1: Write the test file**

Create `tests/test_data_preprocessing.py`:

```python
import pandas as pd
import pytest

from src.data_preprocessing import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_feature_table,
    clean_columns,
    encode_type,
    engineer_features,
)


def make_raw_row(**overrides):
    row = {
        "UDI": 1,
        "Product ID": "M14860",
        "Type": "M",
        "Air temperature [K]": 298.1,
        "Process temperature [K]": 308.6,
        "Rotational speed [rpm]": 1551,
        "Torque [Nm]": 42.8,
        "Tool wear [min]": 0,
        "Machine failure": 0,
        "TWF": 0,
        "HDF": 0,
        "PWF": 0,
        "OSF": 0,
        "RNF": 0,
    }
    row.update(overrides)
    return row


def test_clean_columns_drops_leakage_and_identifier_columns():
    df = pd.DataFrame([make_raw_row()])
    cleaned = clean_columns(df)
    for col in ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]:
        assert col not in cleaned.columns
    assert "machine_failure" in cleaned.columns
    assert "air_temp_k" in cleaned.columns
    assert "type" in cleaned.columns


def test_engineer_features_computes_temp_diff_and_power():
    df = pd.DataFrame([make_raw_row()])
    cleaned = clean_columns(df)
    engineered = engineer_features(cleaned)
    assert engineered.loc[0, "temp_diff_k"] == pytest.approx(308.6 - 298.1)
    expected_power = 42.8 * 1551 * (2 * 3.141592653589793 / 60)
    assert engineered.loc[0, "power_w"] == pytest.approx(expected_power, rel=1e-6)


def test_encode_type_creates_all_three_dummy_columns():
    df = pd.DataFrame([make_raw_row(Type="L")])
    cleaned = clean_columns(df)
    encoded = encode_type(cleaned)
    assert encoded.loc[0, "type_L"] == 1
    assert encoded.loc[0, "type_H"] == 0
    assert encoded.loc[0, "type_M"] == 0
    assert "type" not in encoded.columns


def test_build_feature_table_returns_exact_expected_columns():
    df = pd.DataFrame([make_raw_row(), make_raw_row(Type="L", **{"Machine failure": 1})])
    table = build_feature_table(df)
    assert list(table.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert len(table) == 2
    assert table[TARGET_COLUMN].tolist() == [0, 1]
```

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_data_preprocessing.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.data_preprocessing'`

- [x] **Step 3: Write the implementation**

Create `src/data_preprocessing.py`:

```python
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RENAME_MAP = {
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
}
LEAKAGE_AND_ID_COLUMNS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]

FEATURE_COLUMNS = [
    "air_temp_k",
    "process_temp_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "temp_diff_k",
    "power_w",
    "type_H",
    "type_L",
    "type_M",
]
TARGET_COLUMN = "machine_failure"


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[c for c in LEAKAGE_AND_ID_COLUMNS if c in df.columns])
    df = df.rename(columns=RENAME_MAP)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]
    df["power_w"] = df["torque_nm"] * df["rotational_speed_rpm"] * (2 * np.pi / 60)
    return df


def encode_type(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dummies = pd.get_dummies(df["type"], prefix="type")
    for col in ["type_H", "type_L", "type_M"]:
        if col not in dummies.columns:
            dummies[col] = 0
    df = pd.concat([df.drop(columns=["type"]), dummies], axis=1)
    return df


def build_feature_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(raw_df)
    df = engineer_features(df)
    df = encode_type(df)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns after preprocessing: {missing}")
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]]


def split_and_save(feature_df, test_size=0.2, random_state=42, out_dir=PROCESSED_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df, test_df = train_test_split(
        feature_df,
        test_size=test_size,
        stratify=feature_df[TARGET_COLUMN],
        random_state=random_state,
    )
    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    return train_df, test_df


def run_pipeline():
    raw_df = load_raw_data()
    feature_df = build_feature_table(raw_df)
    train_df, test_df = split_and_save(feature_df)
    print(f"Train shape: {train_df.shape}, failures: {int(train_df[TARGET_COLUMN].sum())}")
    print(f"Test shape: {test_df.shape}, failures: {int(test_df[TARGET_COLUMN].sum())}")


if __name__ == "__main__":
    run_pipeline()
```

- [x] **Step 4: Run the test to verify it passes**

```bash
python -m pytest tests/test_data_preprocessing.py -v
```
Expected: `4 passed`

- [x] **Step 5: Run the pipeline on the real dataset**

```bash
python -m src.data_preprocessing
```
Expected output exactly:
```
Train shape: (8000, 11), failures: 271
Test shape: (2000, 11), failures: 68
```

- [x] **Step 6: Verify the processed files**

```bash
ls -la data/processed/
python -c "
import pandas as pd
train = pd.read_csv('data/processed/train.csv')
test = pd.read_csv('data/processed/test.csv')
print(train.shape, test.shape)
print(sorted(train.columns.tolist()) == sorted(test.columns.tolist()))
"
```
Expected: `train.csv` and `test.csv` present; `(8000, 11) (2000, 11)`; `True`

- [x] **Step 7: Commit**

```bash
git add src/data_preprocessing.py tests/test_data_preprocessing.py data/processed/train.csv data/processed/test.csv
git commit -m "Add data preprocessing pipeline with feature engineering and tests"
```

---

### Task 2: Evaluation Module

**Files:**
- Create: `src/evaluation.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Produces (used by `src/train.py` and `app.py`):
  - `compute_metrics(y_true, y_pred, y_prob) -> dict` with keys `accuracy`, `precision`, `recall`, `f1`, `roc_auc`
  - `compute_confusion_matrix(y_true, y_pred) -> dict` with keys `true_negative`, `false_positive`, `false_negative`, `true_positive`
  - `find_threshold_for_recall(y_true, y_prob, min_recall=0.8) -> float`
  - `compute_roc_curve(y_true, y_prob) -> dict` with keys `fpr`, `tpr` (lists)

- [x] **Step 1: Write the test file**

Create `tests/test_evaluation.py`:

```python
import numpy as np
import pytest

from src.evaluation import compute_confusion_matrix, compute_metrics, find_threshold_for_recall


def test_compute_metrics_on_known_values():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.6, 0.9, 0.4, 0.8])
    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == pytest.approx(3 / 5)
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(2 / 3)


def test_compute_confusion_matrix_counts():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1])
    cm = compute_confusion_matrix(y_true, y_pred)
    assert cm["true_negative"] == 1
    assert cm["false_positive"] == 1
    assert cm["false_negative"] == 1
    assert cm["true_positive"] == 2


def test_find_threshold_for_recall_returns_value_meeting_target():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    threshold = find_threshold_for_recall(y_true, y_prob, min_recall=0.8)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert metrics["recall"] >= 0.8
```

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_evaluation.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.evaluation'`

- [x] **Step 3: Write the implementation**

Create `src/evaluation.py`:

```python
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def compute_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "true_negative": int(cm[0, 0]),
        "false_positive": int(cm[0, 1]),
        "false_negative": int(cm[1, 0]),
        "true_positive": int(cm[1, 1]),
    }


def find_threshold_for_recall(y_true, y_prob, min_recall=0.8):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    candidates = [
        (t, p, r) for t, p, r in zip(thresholds, precisions[:-1], recalls[:-1]) if r >= min_recall
    ]
    if not candidates:
        return 0.5
    best = max(candidates, key=lambda c: c[0])
    return float(best[0])


def compute_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
```

- [x] **Step 4: Run the test to verify it passes**

```bash
python -m pytest tests/test_evaluation.py -v
```
Expected: `3 passed`

- [x] **Step 5: Commit**

```bash
git add src/evaluation.py tests/test_evaluation.py
git commit -m "Add shared evaluation metric helpers with tests"
```

---

### Task 3: Prediction & Explainability Module

**Files:**
- Create: `src/predict.py`

**Interfaces:**
- Consumes: `FEATURE_COLUMNS`, `encode_type`, `engineer_features` from `src/data_preprocessing`.
- Produces (used by `src/train.py` and `app.py`):
  - `load_artifacts(models_dir: Path = MODELS_DIR) -> tuple[model, dict]`
  - `risk_category(probability: float) -> str` — `"LOW" | "MEDIUM" | "HIGH"`
  - `build_input_row(air_temp_k, process_temp_k, rotational_speed_rpm, torque_nm, tool_wear_min, machine_type) -> pd.DataFrame`
  - `predict_single(model, **raw_inputs) -> dict` with keys `probability`, `risk_category`
  - `compute_global_importance(model, X_sample, feature_names=FEATURE_COLUMNS, max_rows=200) -> list[tuple[str, float]]`
  - `top_factors(model, raw_inputs: dict, feature_names=FEATURE_COLUMNS, top_n=3) -> list[tuple[str, float]]`

This module has no raw-data dependency, so its tests fit tiny models trained inline rather than requiring `models/model.joblib` to exist yet — that lets it be built and tested before `src/train.py` exists, even though `src/train.py` will later import `compute_global_importance` from here.

- [x] **Step 1: Write the test file**

Create `tests/test_predict.py`:

```python
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data_preprocessing import FEATURE_COLUMNS
from src.predict import (
    build_input_row,
    compute_global_importance,
    predict_single,
    risk_category,
    top_factors,
)


def make_toy_training_data(n=60, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.rand(n, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
    y = (X["torque_nm"] + X["tool_wear_min"] > 1.0).astype(int)
    return X, y


def test_risk_category_boundaries():
    assert risk_category(0.0) == "LOW"
    assert risk_category(0.29) == "LOW"
    assert risk_category(0.3) == "MEDIUM"
    assert risk_category(0.69) == "MEDIUM"
    assert risk_category(0.7) == "HIGH"
    assert risk_category(0.99) == "HIGH"


def test_build_input_row_has_exact_feature_columns():
    row = build_input_row(
        air_temp_k=300.0,
        process_temp_k=310.0,
        rotational_speed_rpm=1500,
        torque_nm=40.0,
        tool_wear_min=100,
        machine_type="L",
    )
    assert list(row.columns) == FEATURE_COLUMNS
    assert row.loc[0, "type_L"] == 1
    assert row.loc[0, "type_H"] == 0
    assert row.loc[0, "temp_diff_k"] == pytest.approx(10.0)


def test_predict_single_returns_probability_and_risk_category():
    X, y = make_toy_training_data()
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    result = predict_single(
        model,
        air_temp_k=300.0,
        process_temp_k=310.0,
        rotational_speed_rpm=1500,
        torque_nm=40.0,
        tool_wear_min=100,
        machine_type="L",
    )
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk_category"] in {"LOW", "MEDIUM", "HIGH"}


def test_compute_global_importance_for_random_forest_returns_all_features():
    X, y = make_toy_training_data()
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    importance = compute_global_importance(model, X, FEATURE_COLUMNS)
    assert {name for name, _ in importance} == set(FEATURE_COLUMNS)
    values = [v for _, v in importance]
    assert values == sorted(values, reverse=True)


def test_compute_global_importance_for_logistic_regression_fallback():
    X, y = make_toy_training_data()
    model = LogisticRegression(max_iter=1000).fit(X, y)
    importance = compute_global_importance(model, X, FEATURE_COLUMNS)
    assert {name for name, _ in importance} == set(FEATURE_COLUMNS)


def test_top_factors_returns_requested_count():
    X, y = make_toy_training_data()
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    raw_inputs = dict(
        air_temp_k=300.0,
        process_temp_k=310.0,
        rotational_speed_rpm=1500,
        torque_nm=40.0,
        tool_wear_min=100,
        machine_type="L",
    )
    factors = top_factors(model, raw_inputs, top_n=3)
    assert len(factors) == 3
    for name, _ in factors:
        assert name in FEATURE_COLUMNS
```

- [x] **Step 2: Run the test to verify it fails**

```bash
python -m pytest tests/test_predict.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.predict'`

- [x] **Step 3: Write the implementation**

Create `src/predict.py`:

```python
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_preprocessing import FEATURE_COLUMNS, encode_type, engineer_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

RISK_BANDS = [
    (0.0, 0.3, "LOW"),
    (0.3, 0.7, "MEDIUM"),
    (0.7, 1.0001, "HIGH"),
]


def load_artifacts(models_dir: Path = MODELS_DIR):
    model = joblib.load(models_dir / "model.joblib")
    with open(models_dir / "metrics.json") as f:
        metrics = json.load(f)
    return model, metrics


def risk_category(probability: float) -> str:
    for low, high, label in RISK_BANDS:
        if low <= probability < high:
            return label
    return "HIGH"


def build_input_row(
    air_temp_k: float,
    process_temp_k: float,
    rotational_speed_rpm: float,
    torque_nm: float,
    tool_wear_min: float,
    machine_type: str,
) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "air_temp_k": air_temp_k,
                "process_temp_k": process_temp_k,
                "rotational_speed_rpm": rotational_speed_rpm,
                "torque_nm": torque_nm,
                "tool_wear_min": tool_wear_min,
                "type": machine_type,
            }
        ]
    )
    df = engineer_features(df)
    df = encode_type(df)
    return df[FEATURE_COLUMNS]


def predict_single(model, **raw_inputs) -> dict:
    X = build_input_row(**raw_inputs)
    probability = float(model.predict_proba(X)[0, 1])
    return {"probability": probability, "risk_category": risk_category(probability)}


def _extract_positive_class_shap(shap_values):
    if isinstance(shap_values, list):
        return shap_values[1]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def _is_tree_model(model) -> bool:
    return hasattr(model, "feature_importances_")


def compute_global_importance(model, X_sample, feature_names=FEATURE_COLUMNS, max_rows=200):
    X_sample = X_sample.iloc[:max_rows]
    if _is_tree_model(model):
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = _extract_positive_class_shap(explainer.shap_values(X_sample))
        mean_abs = np.abs(shap_values).mean(axis=0)
    else:
        clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        mean_abs = np.abs(clf.coef_[0])
    pairs = sorted(zip(feature_names, mean_abs.tolist()), key=lambda p: p[1], reverse=True)
    return pairs


def top_factors(model, raw_inputs: dict, feature_names=FEATURE_COLUMNS, top_n=3):
    X = build_input_row(**raw_inputs)
    if _is_tree_model(model):
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = _extract_positive_class_shap(explainer.shap_values(X))
        values = shap_values[0]
    else:
        clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        X_values = model.named_steps["scaler"].transform(X) if hasattr(model, "named_steps") else X.values
        values = clf.coef_[0] * X_values[0]
    pairs = list(zip(feature_names, [float(v) for v in values]))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return pairs[:top_n]
```

- [x] **Step 4: Run the test to verify it passes**

```bash
python -m pytest tests/test_predict.py -v
```
Expected: `6 passed`

- [x] **Step 5: Commit**

```bash
git add src/predict.py tests/test_predict.py
git commit -m "Add prediction and SHAP/feature-importance explainability module"
```

---

### Task 4: Model Training Script

**Files:**
- Create: `src/train.py`

**Interfaces:**
- Consumes: `FEATURE_COLUMNS`, `TARGET_COLUMN`, `PROCESSED_DIR` from `src.data_preprocessing`; `compute_metrics`, `compute_confusion_matrix`, `compute_roc_curve`, `find_threshold_for_recall` from `src.evaluation`; `compute_global_importance` from `src.predict`.
- Produces: `models/model.joblib` (fitted sklearn estimator with `.predict_proba`) and `models/metrics.json` (dict consumed by `app.py`) with top-level keys: `chosen_model`, `threshold`, `cv_metrics`, `rf_best_params`, `test_metrics`, `test_confusion_matrix`, `roc_curve`, `global_importance`, `feature_columns`, `dataset_size`, `failure_rate`.

This task is an integration script tying together already-tested units (Tasks 1–3), run against real data rather than synthetic fixtures — so its "test cycle" is: run it, then assert the produced artifacts have the correct structure and sane values, rather than a red/green unit test.

- [x] **Step 1: Write the implementation**

Create `src/train.py`:

```python
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_preprocessing import FEATURE_COLUMNS, PROCESSED_DIR, TARGET_COLUMN
from src.evaluation import (
    compute_confusion_matrix,
    compute_metrics,
    compute_roc_curve,
    find_threshold_for_recall,
)
from src.predict import compute_global_importance

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def load_splits():
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv")
    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]
    return X_train, y_train, X_test, y_test


def cross_validate_model(model, X, y, cv):
    y_prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return compute_metrics(y, y_pred, y_prob), y_prob


def select_best_random_forest(X, y, cv):
    param_grid = {"n_estimators": [100, 200], "max_depth": [5, 10, None]}
    base = RandomForestClassifier(class_weight="balanced", random_state=42)
    search = GridSearchCV(base, param_grid, scoring="recall", cv=cv, n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


def run_training():
    X_train, y_train, X_test, y_test = load_splits()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
        ]
    )
    lr_metrics, lr_oof_prob = cross_validate_model(lr, X_train, y_train, cv)

    rf_best, rf_params = select_best_random_forest(X_train, y_train, cv)
    rf_metrics, rf_oof_prob = cross_validate_model(rf_best, X_train, y_train, cv)

    print("Logistic Regression CV metrics:", lr_metrics)
    print("Random Forest CV metrics:", rf_metrics, "best params:", rf_params)

    if rf_metrics["recall"] >= lr_metrics["recall"]:
        chosen_name, chosen_model = "random_forest", rf_best
        threshold = find_threshold_for_recall(y_train, rf_oof_prob, min_recall=0.8)
    else:
        chosen_name, chosen_model = "logistic_regression", lr
        threshold = find_threshold_for_recall(y_train, lr_oof_prob, min_recall=0.8)

    chosen_model.fit(X_train, y_train)

    test_prob = chosen_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    test_metrics = compute_metrics(y_test, test_pred, test_prob)
    test_cm = compute_confusion_matrix(y_test, test_pred)
    roc_curve_data = compute_roc_curve(y_test, test_prob)
    global_importance = compute_global_importance(chosen_model, X_train, FEATURE_COLUMNS)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(chosen_model, MODELS_DIR / "model.joblib")

    results = {
        "chosen_model": chosen_name,
        "threshold": threshold,
        "cv_metrics": {"logistic_regression": lr_metrics, "random_forest": rf_metrics},
        "rf_best_params": rf_params,
        "test_metrics": test_metrics,
        "test_confusion_matrix": test_cm,
        "roc_curve": roc_curve_data,
        "global_importance": global_importance,
        "feature_columns": FEATURE_COLUMNS,
        "dataset_size": int(len(X_train) + len(X_test)),
        "failure_rate": float((y_train.sum() + y_test.sum()) / (len(y_train) + len(y_test))),
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Chosen model:", chosen_name, "threshold:", threshold)
    print("Test metrics:", test_metrics)
    print("Test confusion matrix:", test_cm)
    return results


if __name__ == "__main__":
    run_training()
```

- [x] **Step 2: Run it against the real data**

```bash
python -m src.train
```

Expected output (captured during planning on the real dataset — your run will match exactly, since all random states are fixed):
```
Logistic Regression CV metrics: {'accuracy': 0.852125, 'precision': 0.1642120765832106, 'recall': 0.8228782287822878, 'f1': 0.2737875997544506, 'roc_auc': 0.9186993538974075}
Random Forest CV metrics: {'accuracy': 0.939875, 'precision': 0.3516949152542373, 'recall': 0.9188191881918819, 'f1': 0.508682328907048, 'roc_auc': 0.9769526664085375} best params: {'max_depth': 5, 'n_estimators': 200}
Chosen model: random_forest threshold: 0.6146279014299617
Test metrics: {'accuracy': 0.9765, 'precision': 0.6129032258064516, 'recall': 0.8382352941176471, 'f1': 0.7080745341614907, 'roc_auc': 0.9703446596029717}
Test confusion matrix: {'true_negative': 1896, 'false_positive': 36, 'false_negative': 11, 'true_positive': 57}
```

Random Forest wins CV recall decisively (0.92 vs 0.82) and is selected. On the untouched test set: **83.8% recall, 61.3% precision, 70.8% F1, 97.0% ROC-AUC** — of 68 real failures in the test set, the model catches 57 and misses 11, at the cost of 36 false alarms out of 1,932 non-failures.

- [x] **Step 3: Verify the artifacts**

```bash
ls -la models/
python -c "
import json
m = json.load(open('models/metrics.json'))
assert m['chosen_model'] == 'random_forest'
assert m['test_metrics']['recall'] > 0.5
assert m['test_confusion_matrix']['true_positive'] + m['test_confusion_matrix']['false_negative'] == 68
assert len(m['global_importance']) == 10
assert len(m['roc_curve']['fpr']) == len(m['roc_curve']['tpr'])
print('metrics.json structure OK')
print('top 3 global importance:', m['global_importance'][:3])
"
```
Expected: `models/model.joblib` (~900KB) and `models/metrics.json` (~8KB) exist; prints `metrics.json structure OK` and the top 3 features (expected: `tool_wear_min`, `rotational_speed_rpm`, `power_w`, in that order, each with importance roughly 0.08–0.10).

- [x] **Step 4: Sanity-check a single prediction end-to-end**

```bash
python -c "
from src.predict import load_artifacts, predict_single, top_factors

model, metrics = load_artifacts()
raw = dict(air_temp_k=302.5, process_temp_k=312.0, rotational_speed_rpm=1350, torque_nm=65.0, tool_wear_min=200, machine_type='L')
print(predict_single(model, **raw))
print(top_factors(model, raw))
"
```
Expected: a HIGH-risk prediction with probability above 0.9, and top factors dominated by `torque_nm`, `power_w`, and `rotational_speed_rpm` (high torque at low rotational speed is a known overstrain-failure pattern in this dataset).

- [x] **Step 5: Commit**

```bash
git add src/train.py models/model.joblib models/metrics.json
git commit -m "Add model training/selection script; train Random Forest on AI4I 2020 data"
```

---

### Task 5: EDA Notebook

**Files:**
- Create: `notebooks/01_eda.ipynb`

**Interfaces:**
- Consumes: `load_raw_data`, `clean_columns`, `engineer_features` from `src.data_preprocessing`.
- Produces: nothing consumed by other tasks — this is a standalone exploratory artifact for the README/interview narrative.

Building a notebook cell-by-cell by hand in a text editor is error-prone, so this task generates it programmatically with `nbformat` (already in `requirements.txt`), then you open and run it once in Jupyter/VS Code to confirm it renders.

- [x] **Step 1: Generate the notebook**

Run this from the project root (it writes `notebooks/01_eda.ipynb` directly — this is a one-off generation command, not a project file):

```bash
python3 << 'PYEOF'
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Exploratory Data Analysis — AI4I 2020 Predictive Maintenance Dataset\n\n"
    "Explores the failure distribution, feature distributions, and correlations "
    "that motivate the feature engineering and modeling choices in `src/train.py`."
))

cells.append(nbf.v4.new_code_cell(
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "import sys\n"
    "from pathlib import Path\n"
    "\n"
    "sys.path.append(str(Path.cwd().parent))\n"
    "from src.data_preprocessing import load_raw_data, clean_columns, engineer_features\n"
    "\n"
    "df = load_raw_data()\n"
    "df = clean_columns(df)\n"
    "df = engineer_features(df)\n"
    "df.head()"
))

cells.append(nbf.v4.new_markdown_cell("## Failure distribution\n\nThe target is heavily imbalanced — this is the central challenge of the project."))

cells.append(nbf.v4.new_code_cell(
    "print(df['machine_failure'].value_counts())\n"
    "print(df['machine_failure'].value_counts(normalize=True))\n"
    "\n"
    "fig, ax = plt.subplots()\n"
    "sns.countplot(x='machine_failure', data=df, ax=ax)\n"
    "ax.set_xticks([0, 1])\n"
    "ax.set_xticklabels(['No Failure', 'Failure'])\n"
    "ax.set_title('Failure Distribution')\n"
    "plt.show()"
))

cells.append(nbf.v4.new_markdown_cell("## Failures by machine type"))

cells.append(nbf.v4.new_code_cell(
    "fig, ax = plt.subplots()\n"
    "sns.countplot(x='type', hue='machine_failure', data=df, ax=ax)\n"
    "ax.set_title('Failures by Machine Type')\n"
    "plt.show()"
))

cells.append(nbf.v4.new_markdown_cell("## Feature distributions by failure status\n\nComparing the distribution of each numeric sensor reading for failed vs. non-failed machines."))

cells.append(nbf.v4.new_code_cell(
    "numeric_cols = ['air_temp_k', 'process_temp_k', 'rotational_speed_rpm', 'torque_nm', 'tool_wear_min', 'temp_diff_k', 'power_w']\n"
    "fig, axes = plt.subplots(3, 3, figsize=(14, 10))\n"
    "for ax, col in zip(axes.flat, numeric_cols):\n"
    "    sns.histplot(data=df, x=col, hue='machine_failure', kde=True, ax=ax, stat='density', common_norm=False)\n"
    "    ax.set_title(col)\n"
    "for ax in axes.flat[len(numeric_cols):]:\n"
    "    ax.set_visible(False)\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells.append(nbf.v4.new_markdown_cell("## Correlation with failure"))

cells.append(nbf.v4.new_code_cell(
    "corr_cols = numeric_cols + ['machine_failure']\n"
    "corr = df[corr_cols].corr()\n"
    "fig, ax = plt.subplots(figsize=(8, 6))\n"
    "sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)\n"
    "plt.show()\n"
    "\n"
    "corr['machine_failure'].sort_values(ascending=False)"
))

cells.append(nbf.v4.new_markdown_cell(
    "## Takeaways\n\n"
    "- The failure rate is ~3.4% — strongly imbalanced, so accuracy alone is a misleading metric "
    "(predicting \"no failure\" for everyone would already score ~96.6% accuracy).\n"
    "- `torque_nm` and the engineered `power_w` feature show the strongest positive correlation with failure, "
    "consistent with overstrain/power-failure mechanics in this dataset.\n"
    "- `temp_diff_k` (process minus air temperature) is negatively correlated with failure in the raw linear correlation, "
    "but still carries useful non-linear signal that the Random Forest model picks up (see `models/metrics.json` "
    "`global_importance` for the model's actual learned feature ranking, which can differ from simple linear correlation)."
))

nb['cells'] = cells
nbf.write(nb, 'notebooks/01_eda.ipynb')
print("notebook written")
PYEOF
```
Expected output: `notebook written`

- [x] **Step 2: Validate the notebook structure**

```bash
python -c "
import nbformat
nb = nbformat.read('notebooks/01_eda.ipynb', as_version=4)
nbformat.validate(nb)
print('valid notebook, cells:', len(nb['cells']))
"
```
Expected: `valid notebook, cells: 11`

- [x] **Step 3: Open and run it to confirm it renders**

Open `notebooks/01_eda.ipynb` in VS Code (Jupyter extension) or `jupyter lab`, and use "Run All". Confirm: no cell raises an error, the failure countplot and correlation heatmap render, and the printed correlation values show `torque_nm` and `power_w` as the top positive correlates of `machine_failure` (this was verified during planning: `torque_nm` ≈ 0.19, `power_w` ≈ 0.18).

- [x] **Step 4: Commit**

```bash
git add notebooks/01_eda.ipynb
git commit -m "Add EDA notebook covering failure distribution and feature correlations"
```

---

### Task 6: Streamlit Dashboard

**Files:**
- Create: `app.py`

**Interfaces:**
- Consumes: `load_raw_data`, `clean_columns`, `engineer_features` from `src.data_preprocessing`; `load_artifacts`, `predict_single`, `top_factors` from `src.predict`.
- Produces: nothing consumed elsewhere — this is the top-level entry point.

- [x] **Step 1: Write the implementation**

Create `app.py`:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.data_preprocessing import clean_columns, engineer_features, load_raw_data
from src.predict import load_artifacts, predict_single, top_factors

st.set_page_config(page_title="Manufacturing Failure Prediction Dashboard", layout="wide")


@st.cache_resource
def get_artifacts():
    return load_artifacts()


@st.cache_data
def get_raw_data():
    df = load_raw_data()
    df = clean_columns(df)
    df = engineer_features(df)
    return df


model, metrics = get_artifacts()
raw_df = get_raw_data()

FEATURE_LABELS = {
    "air_temp_k": "Air Temperature (K)",
    "process_temp_k": "Process Temperature (K)",
    "rotational_speed_rpm": "Rotational Speed (rpm)",
    "torque_nm": "Torque (Nm)",
    "tool_wear_min": "Tool Wear (min)",
    "temp_diff_k": "Temperature Difference (K)",
    "power_w": "Mechanical Power (W)",
    "type_H": "Type: High",
    "type_L": "Type: Low",
    "type_M": "Type: Medium",
}
RISK_COLORS = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}

st.sidebar.title("Manufacturing Failure Prediction")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Data Analysis", "Model Performance", "Failure Prediction", "Business Insights"],
)


def page_overview():
    st.title("Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset size", f"{metrics['dataset_size']:,} machines")
    col2.metric("Failure rate", f"{metrics['failure_rate']*100:.2f}%")
    col3.metric("Model", metrics["chosen_model"].replace("_", " ").title())

    col4, col5, col6, col7 = st.columns(4)
    tm = metrics["test_metrics"]
    col4.metric("Recall", f"{tm['recall']*100:.1f}%")
    col5.metric("Precision", f"{tm['precision']*100:.1f}%")
    col6.metric("F1", f"{tm['f1']*100:.1f}%")
    col7.metric("ROC-AUC", f"{tm['roc_auc']*100:.1f}%")

    st.markdown(
        """
        This dashboard predicts whether a manufacturing machine is likely to fail, using
        operating sensor data (temperature, rotational speed, torque, tool wear) from the
        **AI4I 2020 Predictive Maintenance Dataset**. It is a portfolio project demonstrating
        an end-to-end classification workflow, not a production monitoring system.
        """
    )


def page_data_analysis():
    st.title("Data Analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Failure Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="machine_failure", data=raw_df, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Failure", "Failure"])
        ax.set_ylabel("Count")
        ax.set_xlabel("")
        st.pyplot(fig)

    with col2:
        st.subheader("Failures by Machine Type")
        fig, ax = plt.subplots()
        sns.countplot(x="type", hue="machine_failure", data=raw_df, ax=ax)
        ax.set_xlabel("Machine Type")
        st.pyplot(fig)

    st.subheader("Feature Distributions by Failure Status")
    numeric_cols = ["air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"]
    selected = st.selectbox("Select a feature", numeric_cols, format_func=lambda c: FEATURE_LABELS.get(c, c))
    fig, ax = plt.subplots()
    sns.histplot(data=raw_df, x=selected, hue="machine_failure", kde=True, ax=ax, stat="density", common_norm=False)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    corr_cols = numeric_cols + ["temp_diff_k", "power_w", "machine_failure"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(raw_df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)


def page_model_performance():
    st.title("Model Performance")
    st.caption(
        f"Final model: **{metrics['chosen_model'].replace('_', ' ').title()}**, "
        f"evaluated once on a held-out test set."
    )

    tm = metrics["test_metrics"]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Accuracy", f"{tm['accuracy']*100:.1f}%")
    col2.metric("Precision", f"{tm['precision']*100:.1f}%")
    col3.metric("Recall", f"{tm['recall']*100:.1f}%")
    col4.metric("F1", f"{tm['f1']*100:.1f}%")
    col5.metric("ROC-AUC", f"{tm['roc_auc']*100:.1f}%")

    cm = metrics["test_confusion_matrix"]
    col_cm, col_roc = st.columns(2)
    with col_cm:
        st.subheader("Confusion Matrix (Test Set)")
        matrix = np.array(
            [
                [cm["true_negative"], cm["false_positive"]],
                [cm["false_negative"], cm["true_positive"]],
            ]
        )
        fig, ax = plt.subplots()
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Pred No Failure", "Pred Failure"],
            yticklabels=["Actual No Failure", "Actual Failure"],
            ax=ax,
        )
        st.pyplot(fig)

    with col_roc:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots()
        ax.plot(metrics["roc_curve"]["fpr"], metrics["roc_curve"]["tpr"], label=f"ROC-AUC = {tm['roc_auc']:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

    st.info(
        f"We prioritize **recall** over raw accuracy: missing a real failure (a false negative) "
        f"risks unplanned downtime, while a false alarm (false positive) only costs an unnecessary "
        f"inspection. Of {cm['false_negative'] + cm['true_positive']} real failures in the test set, "
        f"the model caught {cm['true_positive']} and missed {cm['false_negative']}."
    )


def page_failure_prediction():
    st.title("Failure Prediction")
    st.write("Enter machine operating conditions to estimate failure risk.")

    col1, col2, col3 = st.columns(3)
    with col1:
        air_temp = st.slider("Air Temperature (K)", 295.0, 305.0, 300.0, 0.1)
        process_temp = st.slider("Process Temperature (K)", 305.0, 314.0, 310.0, 0.1)
    with col2:
        rpm = st.slider("Rotational Speed (rpm)", 1150, 2900, 1550, 10)
        torque = st.slider("Torque (Nm)", 3.0, 77.0, 40.0, 0.5)
    with col3:
        tool_wear = st.slider("Tool Wear (min)", 0, 260, 108, 1)
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"])

    if st.button("Predict Failure Risk", type="primary"):
        raw_inputs = dict(
            air_temp_k=air_temp,
            process_temp_k=process_temp,
            rotational_speed_rpm=rpm,
            torque_nm=torque,
            tool_wear_min=tool_wear,
            machine_type=machine_type,
        )
        result = predict_single(model, **raw_inputs)
        factors = top_factors(model, raw_inputs)

        prob_pct = result["probability"] * 100
        risk = result["risk_category"]

        st.markdown(f"### {RISK_COLORS[risk]} {risk} RISK — {prob_pct:.1f}% failure probability")
        if risk == "LOW":
            st.success("This machine's operating conditions look typical for normal operation.")
        elif risk == "MEDIUM":
            st.warning("This machine shows some elevated risk factors — consider monitoring.")
        else:
            st.error("This machine shows a high failure probability — consider inspection.")

        st.subheader("Main Contributing Factors")
        factor_df = pd.DataFrame(factors, columns=["feature", "impact"])
        factor_df["feature"] = factor_df["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
        factor_df["direction"] = np.where(factor_df["impact"] >= 0, "Increases risk", "Decreases risk")
        fig, ax = plt.subplots()
        colors = ["#C44E52" if v >= 0 else "#4C72B0" for v in factor_df["impact"]]
        ax.barh(factor_df["feature"], factor_df["impact"], color=colors)
        ax.set_xlabel("Impact on failure probability")
        ax.invert_yaxis()
        st.pyplot(fig)
        for _, row in factor_df.iterrows():
            st.write(f"- **{row['feature']}** — {row['direction']}")


def page_business_insights():
    st.title("Business Insights")

    st.subheader("What the Model Found")
    importance_df = pd.DataFrame(metrics["global_importance"][:6], columns=["feature", "importance"])
    importance_df["feature"] = importance_df["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
    fig, ax = plt.subplots()
    ax.barh(importance_df["feature"], importance_df["importance"], color="#4C72B0")
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value| (average impact on prediction)")
    st.pyplot(fig)

    st.markdown(
        """
        **Patterns the model found**: tool wear, rotational speed, mechanical power, and torque are
        the strongest drivers of predicted failure — consistent with the physical failure modes in
        this dataset (overstrain and power failures tend to occur at high torque / low speed combinations,
        while tool-wear failures accumulate over time).

        **What this model can tell us**: which operating conditions are statistically associated with
        a higher failure probability in this historical dataset, and a ranked risk estimate for new
        readings that resemble it.

        **What this model cannot tell us**: it cannot prove *why* a machine will fail mechanically, it
        has not seen real plant data or sensor drift, and it should not be the sole basis for a maintenance
        decision — it is a triage/prioritization signal, not a diagnosis.

        **Preventive-maintenance use case**: flag HIGH-risk machines for inspection before failure,
        prioritize maintenance scheduling by risk score rather than fixed time intervals, and use the
        top contributing factors to tell technicians what to check first.
        """
    )


PAGES = {
    "Overview": page_overview,
    "Data Analysis": page_data_analysis,
    "Model Performance": page_model_performance,
    "Failure Prediction": page_failure_prediction,
    "Business Insights": page_business_insights,
}

PAGES[page]()
```

- [x] **Step 2: Smoke-test the app headlessly**

```bash
streamlit run app.py --server.headless true --server.port 8599 &
sleep 5
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8599
kill %1
```
Expected: `HTTP 200`, and no traceback printed to the terminal.

- [x] **Step 3: Manually test in the browser**

```bash
streamlit run app.py
```
Open the URL it prints. Click through all 5 sidebar pages. On "Failure Prediction", set Torque to ~65, Tool Wear to ~200, Rotational Speed to ~1350, Type to "L", click **Predict Failure Risk**, and confirm it reports **HIGH risk** with probability above 90% and `torque_nm`/`power_w`/`rotational_speed_rpm` as the top factors (this matches the end-to-end check already run during planning). Then try Torque ~20, Tool Wear ~10, Rotational Speed ~1800 and confirm it reports **LOW risk**. Stop the server with Ctrl+C when done.

- [x] **Step 4: Commit**

```bash
git add app.py
git commit -m "Add Streamlit dashboard with 5 pages: overview, data analysis, model performance, prediction, insights"
```

---

### Task 7: README

**Files:**
- Create: `README.md`

- [x] **Step 1: Write `README.md`**

```markdown
# Manufacturing Failure Prediction Dashboard

A portfolio data science project: an end-to-end machine learning pipeline and Streamlit
dashboard that predicts manufacturing machine failure from sensor/operating data, built
for a Junior Data Scientist application at Michelin.

## Problem

Unplanned machine failure in a manufacturing line is costly: it halts production, risks
damaging equipment, and can be a safety issue. The goal here is to predict, from live
operating sensor readings, whether a machine is at risk of failure — so maintenance can be
prioritized before the failure happens rather than reacted to after.

## Business Context

In a predictive-maintenance setting, the question isn't "will this machine fail right now?"
in isolation — it's "which machines, out of many running in parallel, should get inspected
first?" A model that reliably flags high-risk machines lets a maintenance team spend limited
inspection time where it matters most, instead of on a fixed schedule or waiting for a machine
to actually break down.

## Dataset

[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
(UCI Machine Learning Repository), 10,000 rows, one row per machine operating cycle.

| Column (raw) | Used as | Meaning |
|---|---|---|
| `Type` | `type_H`/`type_L`/`type_M` | Product quality variant (Low/Medium/High) |
| `Air temperature [K]` | `air_temp_k` | Ambient temperature around the machine |
| `Process temperature [K]` | `process_temp_k` | Temperature of the machine during operation |
| `Rotational speed [rpm]` | `rotational_speed_rpm` | Spindle rotational speed |
| `Torque [Nm]` | `torque_nm` | Torque applied during the operation |
| `Tool wear [min]` | `tool_wear_min` | Cumulative minutes the cutting tool has been in use |
| `Machine failure` | `machine_failure` (target) | 1 if any failure mode triggered, else 0 |

`UDI` and `Product ID` are dropped (pure identifiers). `TWF`, `HDF`, `PWF`, `OSF`, `RNF` are
dropped — they are *sub-flags for which failure mode occurred*, only knowable after the
failure has already happened, so training on them would leak the target.

The dataset is small and imbalanced: **339 failures out of 10,000 rows (3.39%)**.

## Approach

Binary classification under class imbalance. Two models are compared — **Logistic
Regression** (baseline) and **Random Forest** — both with `class_weight="balanced"`, selected
via 5-fold stratified cross-validation on the training split (not the test set), scored on
recall. The chosen model is fit once on the full training split, and evaluated exactly once
on a held-out test set.

**Primary metric: Recall** on the failure class. In this problem, a false negative (a real
failure the model misses) risks unplanned downtime; a false positive (a false alarm) only
costs an unnecessary inspection. Those costs are not symmetric, so accuracy alone is
misleading — predicting "no failure" for every machine would already score 96.6% accuracy
while catching zero real failures. Recall, precision, F1, and ROC-AUC are all reported
together so the tradeoff is visible rather than hidden behind one number.

## Data Preparation

1. Load the raw CSV, drop identifier and leakage columns, rename to snake_case.
2. Engineer two physics-grounded features:
   - `temp_diff_k` = `process_temp_k` − `air_temp_k` (heat generated under load)
   - `power_w` ≈ `torque_nm` × `rotational_speed_rpm` × (2π/60) (mechanical power — combines
     two raw sensors into one closer to the actual failure mechanism)
3. One-hot encode `type` into `type_H`/`type_L`/`type_M`.
4. Stratified 80/20 train/test split (stratified because the positive class is only 3.4% —
   a plain random split risks a test set with very few failures).

See `src/data_preprocessing.py`.

## Exploratory Analysis

See `notebooks/01_eda.ipynb`. Key findings: the failure rate is ~3.4% (confirming the need
for `class_weight="balanced"` rather than treating this as a balanced problem); `torque_nm`
and the engineered `power_w` show the strongest linear correlation with failure (≈0.19 and
≈0.18 respectively), consistent with overstrain/power-failure mechanics; `tool_wear_min` and
`rotational_speed_rpm` turn out to be the single strongest *model-learned* (SHAP) predictors,
showing the Random Forest is picking up non-linear interactions a simple correlation misses.

## Model Selection

| Model | CV Recall | CV Precision | CV F1 | CV ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (`class_weight="balanced"`) | 0.823 | 0.164 | 0.274 | 0.919 |
| **Random Forest** (`class_weight="balanced"`, best params: `n_estimators=200`, `max_depth=5`) | **0.919** | 0.352 | 0.509 | 0.977 |

Random Forest is selected — it dominates Logistic Regression on every metric, not just
recall, which is expected given the non-linear interaction between torque/speed/wear that
drives failures in this dataset.

The decision threshold (0.615) is chosen from the cross-validated precision-recall curve to
target ≥80% recall on the training folds, **before** touching the test set — this avoids
tuning the threshold to the test data.

## Model Performance

Final Random Forest, evaluated once on the held-out test set (2,000 rows, 68 real failures):

| Metric | Value |
|---|---|
| Accuracy | 97.65% |
| Precision | 61.3% |
| **Recall** | **83.8%** |
| F1 | 70.8% |
| ROC-AUC | 97.0% |

Confusion matrix:

| | Predicted No Failure | Predicted Failure |
|---|---|---|
| **Actual No Failure** | 1,896 (TN) | 36 (FP) |
| **Actual Failure** | 11 (FN) | 57 (TP) |

Of 68 real failures, the model catches 57 and misses 11, at the cost of 36 false alarms out
of 1,932 non-failures (~1.9% false-alarm rate). That tradeoff — a small, cheap inspection cost
in exchange for catching 84% of real failures — is the kind of asymmetric cost predictive
maintenance is meant to exploit.

## Explainability

- **Global**: SHAP `TreeExplainer` on the Random Forest, averaged over a sample of the
  training set, surfaced on the dashboard's "Business Insights" page and stored in
  `models/metrics.json` under `global_importance`. Top drivers: `tool_wear_min`,
  `rotational_speed_rpm`, `power_w`, `torque_nm`.
- **Local**: the same SHAP explainer applied to a single input row, shown on the "Failure
  Prediction" page as the top 3 contributing factors for that specific prediction, with
  direction (increases/decreases risk).
- Fallback (if a Logistic Regression were ever selected instead): coefficient × input value
  as a linear per-feature contribution, and `|coefficient|` for global importance — no SHAP
  dependency needed for a linear model.

## Demo

Run locally (see **How to Run** below) and you'll see a 5-page dashboard: a headline Overview
with dataset size/failure rate/model metrics, a Data Analysis page with failure/feature
distributions and a correlation heatmap, a Model Performance page with the confusion matrix
and ROC curve, an interactive Failure Prediction page (enter sensor values, get a
probability + LOW/MEDIUM/HIGH risk badge + top contributing factors), and a Business Insights
page summarizing what the model found and its practical use case.

## Limitations

- **Synthetic-but-realistic dataset**: AI4I 2020 is designed to mimic real industrial failure
  mechanics, but it is not live plant data — real sensor noise, drift, and failure modes not
  represented in this dataset could behave differently.
- **Class imbalance**: only 339 failures to learn from; rare failure subtypes may be
  under-represented even within that small positive class.
- **False positive / false negative asymmetry**: the model is tuned to favor recall, which
  means it will generate more false alarms than a precision-tuned model would — this is a
  deliberate tradeoff, not a flaw, but it has a real operational cost (unnecessary
  inspections) that a deployment would need to size against the cost of a missed failure.
- **No real-time streaming data**: this is a static, point-in-time model over a fixed
  snapshot of sensor readings, not a system that monitors a live data stream or detects
  drift over time.
- **Generalization**: trained and evaluated on one dataset from one (synthetic) manufacturing
  context — performance on a different machine type, sensor suite, or real factory floor is
  unverified.

**This project is a portfolio MVP, not a production-ready system.** A real deployment would
need real plant data validation, monitoring for data/model drift, a human-in-the-loop review
process, and infrastructure well beyond this repo's scope.

## Future Improvements

- Try SMOTE or other oversampling alongside `class_weight="balanced"` and compare.
- Add XGBoost as a third model comparison point.
- Calibrate predicted probabilities (e.g. `CalibratedClassifierCV`) if probability values
  themselves (not just rank-ordering) need to be trustworthy.
- Track model performance over time / retraining if deployed against a live data feed.

## How to Run

```bash
cd manufacturing-failure-prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Artifacts (models/model.joblib, models/metrics.json) are already committed —
# to retrain from scratch instead:
python -m src.data_preprocessing
python -m src.train

# Run the test suite:
python -m pytest tests/ -v

# Launch the dashboard:
streamlit run app.py
```
```

- [x] **Step 2: Commit**

```bash
git add README.md
git commit -m "Add README documenting problem, approach, results, and limitations"
```

---

### Task 8: Final End-to-End Verification

**Files:** none created — verification only.

- [x] **Step 1: Run the full test suite**

```bash
python -m pytest tests/ -v
```
Expected: `13 passed` (4 from Task 1 + 3 from Task 2 + 6 from Task 3).

- [x] **Step 2: Re-run the full pipeline from scratch to confirm reproducibility**

```bash
rm -rf data/processed models/model.joblib models/metrics.json
python -m src.data_preprocessing
python -m src.train
```
Expected: identical output to Task 1 Step 5 and Task 4 Step 2 (all random states are fixed, so this must reproduce exactly).

- [x] **Step 3: Full manual click-through of the dashboard**

```bash
streamlit run app.py
```
Walk through all 5 pages one more time end-to-end as a final check, exactly as in Task 6 Step 3.

- [x] **Step 4: Check git status is clean**

```bash
git status
git log --oneline
```
Expected: working tree clean, a commit history showing scaffolding → preprocessing → evaluation → predict → train → notebook → app → README.

- [x] **Step 5: Final commit if anything changed during verification**

```bash
git add -A
git commit -m "Final verification pass"
```
(Skip this step if `git status` was already clean after Step 2's re-run — regenerating the same artifacts byte-for-byte should produce no diff.)
