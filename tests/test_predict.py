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
