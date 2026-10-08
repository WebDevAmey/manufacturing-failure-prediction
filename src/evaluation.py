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
