"""
evaluate.py

Evaluation metrics for logistic regression model
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict, Any, List
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
)


@dataclass(frozen=True)
class EvalResult:
    threshold: float

    # Confusion matrix components
    tn: int
    fp: int
    fn: int
    tp: int

    # Metrics
    precision: float
    recall: float
    accuracy: float

    # Full report text
    report_text: str

    # Capture/lift table
    capture_lift: pd.DataFrame


def threshold_predictions(y_proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (y_proba >= threshold).astype(int)


def compute_capture_lift(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    top_percents: Iterable[float] = (0.05, 0.10, 0.20),
) -> pd.DataFrame:
    df_eval = pd.DataFrame({"y": y_true, "p": y_proba}).sort_values("p", ascending=False)

    total_defaulters = int(df_eval["y"].sum())
    if total_defaulters == 0:
        raise ValueError("No positive class instances in y_true; cannot compute capture/lift.")

    rows: List[Dict[str, Any]] = []
    n = len(df_eval)

    for p in top_percents:
        k = max(int(p * n), 1)
        top_slice = df_eval.iloc[:k]

        captured = int(top_slice["y"].sum())
        capture_rate = captured / total_defaulters
        lift = capture_rate / p

        rows.append(
            {
                "Top %": f"{int(p*100)}%",
                "Captured % of defaulters": round(capture_rate * 100, 2),
                "Lift vs Random": round(lift, 2),
                "Count in Top Slice": k,
            }
        )

    return pd.DataFrame(rows)


def evaluate_classifier(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    threshold: float = 0.5,
    top_percents: Iterable[float] = (0.05, 0.10, 0.20),
) -> EvalResult:
    y_pred = threshold_predictions(y_proba, threshold=threshold)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    report = classification_report(y_true, y_pred, zero_division=0)
    caplift = compute_capture_lift(y_true=y_true, y_proba=y_proba, top_percents=top_percents)

    return EvalResult(
        threshold=threshold,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        precision=prec,
        recall=rec,
        accuracy=acc,
        report_text=report,
        capture_lift=caplift,
    )


def save_eval_result(result: EvalResult, out_path: Path) -> None:
    """
    Save evaluation results to JSON file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "threshold": result.threshold,
        "confusion_matrix": {
            "tn": result.tn,
            "fp": result.fp,
            "fn": result.fn,
            "tp": result.tp,
        },
        "metrics": {
            "precision": result.precision,
            "recall": result.recall,
            "accuracy": result.accuracy,
        },
        "capture_lift": result.capture_lift.to_dict(orient="records"),
        "classification_report": result.report_text,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


