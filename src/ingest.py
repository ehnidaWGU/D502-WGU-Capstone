"""
ingest.py

Ingestion utilities for the Home Credit dataset.
This module is intentionally simple and mirrors the original notebook-based ingest.py.
"""

from pathlib import Path
import pandas as pd

from src.paths import get_paths


def get_raw_dir() -> Path:
    """
    Returns the path to the raw dataset directory.
    """
    paths = get_paths()
    return paths.raw_dir


def validate_raw_files(raw_dir: Path) -> None:
    """
    Checks that the expected raw files exist.
    """
    required_files = [
        "application_train.csv",
        "application_test.csv",
        "bureau.csv",
        "bureau_balance.csv",
        "credit_card_balance.csv",
        "installments_payments.csv",
        "POS_CASH_balance.csv",
        "previous_application.csv",
        "HomeCredit_columns_description.csv",
    ]

    missing = [f for f in required_files if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required raw file(s): {missing}")


def load_application_train(path: Path) -> pd.DataFrame:
    """
    Loads application_train.csv and performs basic sanity checks.
    """
    df = pd.read_csv(path)

    if "TARGET" not in df.columns:
        raise ValueError("TARGET column not found in application_train.csv")

    if df.empty:
        raise ValueError("application_train.csv loaded as empty dataframe")

    return df


def save_to_interim(df: pd.DataFrame, filename: str = "application_train.csv") -> Path:
    """
    Saves a dataframe to data/interim.
    """
    paths = get_paths()
    paths.ensure_dirs()

    out_path = paths.interim_dir / filename
    df.to_csv(out_path, index=False)
    return out_path