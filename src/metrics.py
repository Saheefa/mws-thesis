"""Stage 5 metrics: per-class P/R/F1, macro-F1 (primary) and weighted-F1."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support


def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def report(y_true, y_pred, class_names: dict | None = None) -> dict:
    labels = sorted(np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)])))
    p, r, f, n = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    names = class_names or {}
    per_class = pd.DataFrame(
        {"precision": p, "recall": r, "f1": f, "support": n},
        index=[names.get(c, str(c)) for c in labels],
    )
    return {
        "per_class": per_class.round(4).to_dict(orient="index"),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "accuracy": round(float((np.asarray(y_true) == np.asarray(y_pred)).mean()), 4),
    }
