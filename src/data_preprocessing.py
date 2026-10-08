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
