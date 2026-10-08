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
