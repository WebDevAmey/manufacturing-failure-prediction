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
