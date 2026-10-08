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
