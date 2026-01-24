"""
preprocess.py

Preprocessing utilities for Home Credit application_train.csv dataset
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

@dataclass(frozen=True)
class PreprocessArtifacts:
    dropped_columns: List[str]
    numeric_cols: List[str]
    categorical_cols: List[str]
    imputer: SimpleImputer
    ohe: OneHotEncoder
    feature_names: List[str]

    
@dataclass(frozen=True)
class PreprocessResult:
    X_train: np.ndarray
    X_valid: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    artifacts: PreprocessArtifacts

def _make_ohe() -> OneHotEncoder:
    try:
        # sklearn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # sklearn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def preprocess_application_train(
    df: pd.DataFrame,
    *,
    target_col: str = "TARGET",
    missing_threshold: float = 0.50,
    sentinel_days_employed: int = 365243,
    test_size: float = 0.20,
    random_state: int = 69,
) -> PreprocessResult:
    """
    Run preprocessing modeled after preproccessing_02.ipynb.
    Returns PreprocessResult
    """
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' not found in dataframe.")

    df_processing = df.copy()

    # 1) Drop high-missingness columns (modeled after your notebook: computed pre-split)
    missing = df_processing.isna().mean()
    cols_to_drop_mask = missing > missing_threshold
    dropped_columns = list(missing.index[cols_to_drop_mask])
    df_processing = df_processing.loc[:, ~cols_to_drop_mask].copy()

    # 2) Replace sentinel values in DAYS_EMPLOYED
    if "DAYS_EMPLOYED" in df_processing.columns:
        df_processing["DAYS_EMPLOYED"] = df_processing["DAYS_EMPLOYED"].replace(
            sentinel_days_employed, np.nan
        )

    # 3) Train/validation split
    X = df_processing.drop(columns=[target_col])
    y = df_processing[target_col]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # 4) Numeric imputation (median)
    numeric_cols = list(X_train.select_dtypes(include=[np.number]).columns)

    imputer = SimpleImputer(strategy="median")

    X_train = X_train.copy()
    X_valid = X_valid.copy()

    if numeric_cols:
        X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
        X_valid[numeric_cols] = imputer.transform(X_valid[numeric_cols])

    # 5) One-hot encode categoricals (object columns, modeled after notebook)
    categorical_cols = list(X_train.select_dtypes(include=["object"]).columns)

    ohe = _make_ohe()

    if categorical_cols:
        X_train_cat = ohe.fit_transform(X_train[categorical_cols])
        X_valid_cat = ohe.transform(X_valid[categorical_cols])
        cat_feature_names = list(ohe.get_feature_names_out(categorical_cols))
    else:
        # No categoricals present
        X_train_cat = np.empty((len(X_train), 0))
        X_valid_cat = np.empty((len(X_valid), 0))
        cat_feature_names = []

    # 6) Combine numeric + categorical
    if numeric_cols:
        X_train_num = X_train[numeric_cols].to_numpy()
        X_valid_num = X_valid[numeric_cols].to_numpy()
    else:
        X_train_num = np.empty((len(X_train), 0))
        X_valid_num = np.empty((len(X_valid), 0))

    X_train_final = np.hstack([X_train_num, X_train_cat])
    X_valid_final = np.hstack([X_valid_num, X_valid_cat])

    # Sanity check: no NaNs after impute/encode
    if np.isnan(X_train_final).any() or np.isnan(X_valid_final).any():
        raise ValueError("NaNs detected after preprocessing. Check imputation/encoding steps.")

    feature_names = numeric_cols + cat_feature_names

    artifacts = PreprocessArtifacts(
        dropped_columns=dropped_columns,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        imputer=imputer,
        ohe=ohe,
        feature_names=feature_names,
    )

    return PreprocessResult(
        X_train=X_train_final,
        X_valid=X_valid_final,
        y_train=y_train.to_numpy(),
        y_valid=y_valid.to_numpy(),
        artifacts=artifacts,
    )


def transform_new_data(
    df_new: pd.DataFrame,
    artifacts: PreprocessArtifacts,
    *,
    sentinel_days_employed: int = 365243,
) -> np.ndarray:
    """
    Apply fitted preprocessing artifacts to new data.

    This is useful later for scoring application_test.csv or new applicants.

    """
    df_proc = df_new.copy()

    # Drop columns that were dropped during training (ignore if not present)
    drop_existing = [c for c in artifacts.dropped_columns if c in df_proc.columns]
    if drop_existing:
        df_proc = df_proc.drop(columns=drop_existing)

    if "DAYS_EMPLOYED" in df_proc.columns:
        df_proc["DAYS_EMPLOYED"] = df_proc["DAYS_EMPLOYED"].replace(sentinel_days_employed, np.nan)

    # Numeric
    num_cols = artifacts.numeric_cols
    if num_cols:
        X_num = df_proc[num_cols].copy()
        X_num = artifacts.imputer.transform(X_num)
    else:
        X_num = np.empty((len(df_proc), 0))

    # Categoricals
    cat_cols = artifacts.categorical_cols
    if cat_cols:
        X_cat = artifacts.ohe.transform(df_proc[cat_cols])
    else:
        X_cat = np.empty((len(df_proc), 0))

    X_final = np.hstack([X_num, X_cat])

    if np.isnan(X_final).any():
        raise ValueError("NaNs detected after transforming new data.")

    return X_final