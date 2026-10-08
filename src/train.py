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
