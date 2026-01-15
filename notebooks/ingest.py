import pandas as pd
from pathlib import Path

def get_repo_root() -> Path:
    """Gets the repo root to open file"""
    root = Path.cwd()
    while not (root / "data").exists():
        if root.parent == root:
            raise RuntimeError("Could not find repo root (no 'data' directory found)")
        root = root.parent
    return root
    
def get_raw_dir() -> Path:
    """Uses get_repo_root to find raw dir path"""
    root = get_repo_root()
    raw_dir = root / "data" / "raw" / "raw_dataset"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data dir not found: {raw_dir}")
    return raw_dir

def validate_raw_files(raw_dir: Path) -> None:
    """Validates the presence of application_train, raises error if missing"""
    required = {
        "application_train.csv": ["SK_ID_CURR", "TARGET"],
    }

    for fname, required_cols in required.items():
        path = raw_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"Missing required raw file: {path}")

        if path.stat().st_size == 0:
            raise ValueError(f"File is empty: {path}")

        df = pd.read_csv(path, nrows=5)
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"{fname} missing required columns: {missing}")

def load_application_train(path: Path) -> pd.DataFrame:
    """Loads in application_train.csv and validates the basic schema"""
    df = pd.read_csv(path)

    required_cols = {"SK_ID_CURR", "TARGET"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df["SK_ID_CURR"].isna().any():
        raise ValueError("SK_ID_CURR contains nulls")

    duplicates = df["SK_ID_CURR"].duplicated().sum()
    if duplicates > 0:
        raise ValueError(f"SK_ID_CURR is not unique: {duplicates} duplicates found")

    bad_targets = set(df["TARGET"].dropna().unique()) - {0, 1}
    if bad_targets:
        raise ValueError(f"Unexpected TARGET values: {bad_targets}")

    print("Target Distribution:")
    print(df["TARGET"].value_counts(normalize=True))

    return df

