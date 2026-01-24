"""
train.py

Model training fucntions to train Logistic Regression model
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

@dataclass(frozen=True)
class TrainedLogisticRegression:
    model: LogisticRegression
    scaler: StandardScaler  # numeric-only scaler
    num_feature_count: int  # needed to split numeric vs one-hot parts


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    num_feature_count: int,
    max_iter: int = 1000,
    random_state: Optional[int] = None,
) -> TrainedLogisticRegression:
    """
    Train logistic regression model with numeric-only scaling
    """
    X_train_num = X_train[:, :num_feature_count]
    X_train_cat = X_train[:, num_feature_count:]

    scaler = StandardScaler()
    X_train_num_scaled = scaler.fit_transform(X_train_num)

    X_train_scaled = np.hstack([X_train_num_scaled, X_train_cat])

    model = LogisticRegression(max_iter=max_iter, class_weight ="balanced")
    model.fit(X_train_scaled, y_train)

    return TrainedLogisticRegression(model=model, scaler=scaler, num_feature_count=num_feature_count)



def predict_proba_logistic_regression(
    trained: TrainedLogisticRegression,
    X: np.ndarray,
) -> np.ndarray:
    """
    Predict probabilities for the positive class using a trained LR + scaler.
    """
    n = trained.num_feature_count
    X_num = X[:, :n]
    X_cat = X[:, n:]

    X_num_scaled = trained.scaler.transform(X_num)
    X_scaled = np.hstack([X_num_scaled, X_cat])

    return trained.model.predict_proba(X_scaled)[:, 1]