"""Frame and window metrics (PLAN §6). The event metrics come at milestone 6."""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from proctor_assist.labels import CLASSES


def suspicion(probs):
    """The one score that the app reads: 1 - P(no cheat), from 0 to 1 (D10)."""
    return 1 - np.asarray(probs)[:, 0]


def report(targets, probs, majority_class=0):
    """Macro F1, F1 per class, confusion matrix, ROC-AUC for cheat against no cheat, and a baseline.

    The baseline gives `majority_class` for each row. A trained model must beat it (PLAN §5).
    """
    targets, probs = np.asarray(targets), np.asarray(probs)
    labels = list(CLASSES)
    pred = probs.argmax(1)
    cheat = targets > 0
    f1 = f1_score(targets, pred, labels=labels, average=None, zero_division=0)
    baseline = np.full_like(targets, majority_class)
    return {
        "rows": int(len(targets)),
        "macro_f1": float(f1.mean()),
        "f1_per_class": {CLASSES[c]: float(v) for c, v in zip(labels, f1)},
        "accuracy": float((pred == targets).mean()),
        "confusion_matrix_rows_true_cols_pred": confusion_matrix(targets, pred, labels=labels).tolist(),
        # ROC-AUC needs both groups. A split part with one group gives no value.
        "roc_auc_cheat": float(roc_auc_score(cheat, suspicion(probs))) if 0 < cheat.sum() < len(cheat) else None,
        "baseline_macro_f1": float(f1_score(targets, baseline, labels=labels, average="macro", zero_division=0)),
        "baseline_accuracy": float((baseline == targets).mean()),
    }
