
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

from src.paths import get_paths
from src.ingest import get_raw_dir, validate_raw_files, load_application_train, save_to_interim
from src.preprocess import preprocess_application_train
from src.train import train_logistic_regression, predict_proba_logistic_regression
from src.evaluate import evaluate_classifier, save_eval_result


def make_synthetic_smoke_df(n: int = 5000, random_state: int = 69) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    df = pd.DataFrame({
        "SK_ID_CURR": np.arange(100000, 100000 + n),
        "DAYS_BIRTH": -rng.integers(18 * 365, 70 * 365, size=n),
        "DAYS_EMPLOYED": rng.integers(-40 * 365, 0, size=n),
        "AMT_INCOME_TOTAL": rng.normal(150000, 50000, size=n).clip(20000, None),
        "AMT_CREDIT": rng.normal(600000, 200000, size=n).clip(50000, None),
        "AMT_GOODS_PRICE": rng.normal(650000, 220000, size=n).clip(50000, None),
        "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner", "Unemployed"], size=n, p=[0.75, 0.2, 0.05]),
        "NAME_EDUCATION_TYPE": rng.choice(["Secondary", "Higher education"], size=n, p=[0.7, 0.3]),
        "ORGANIZATION_TYPE": rng.choice(["Business Entity", "Realtor", "Government"], size=n),
    })

    # Inject sentinel value sometimes
    sentinel_mask = rng.random(n) < 0.02
    df.loc[sentinel_mask, "DAYS_EMPLOYED"] = 365243

    # Create ~8% default rate with some signal
    base = -2.45
    score = (
        base
        + 0.000002 * (df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"])
        + (df["NAME_INCOME_TYPE"] == "Unemployed").astype(int) * 0.8
        + (df["NAME_INCOME_TYPE"] == "Pensioner").astype(int) * -0.3
    )
    p = 1 / (1 + np.exp(-score))
    df["TARGET"] = (rng.random(n) < p).astype(int)

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Home Credit default risk pipeline.")
    parser.add_argument(
        "--save-interim",
        action="store_true",
        help="If set, save application_train.csv to data/interim/",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for classification (default: 0.5).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="If set, run on a small subset for CI speed (no full-data run).",
    )
    parser.add_argument(
        "--smoke-n",
        type=int,
        default=50_000,
        help="Row count to use for smoke test (default: 50,000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_paths()
    paths.ensure_dirs()

    print("=== D502 Capstone Pipeline شروع ===")
    print(f"Repo root: {paths.root}")
    print(f"Raw dir:   {paths.raw_dir}")
    print(f"Reports:   {paths.reports_dir}")
    print(f"Models:    {paths.models_dir}")

    # 1) Load data (real or synthetic for smoke test)
    raw_dir = get_raw_dir()

    if args.smoke_test and not (raw_dir / "application_train.csv").exists():
        print("SMOKE TEST: raw dataset not found. Using synthetic dataset.")
        df = make_synthetic_smoke_df(n=args.smoke_n, random_state=69)
        print(f"Synthetic dataset shape: {df.shape}")
    else:
        validate_raw_files(raw_dir)
        df = load_application_train(raw_dir / "application_train.csv")
        print(f"Loaded application_train.csv: shape={df.shape}")

    if args.smoke_test:
        # stratification later still works; sample is for CI speed
        df = df.sample(n=min(args.smoke_n, len(df)), random_state=69)
        print(f"SMOKE TEST enabled. Using subset: shape={df.shape}")
    print(f"Default rate in run: {df['TARGET'].mean():.4%}")

    if args.save_interim:
        out_path = save_to_interim(df, filename="application_train.csv")
        print(f"Saved interim CSV to: {out_path}")

    # 2) Preprocess
    prep = preprocess_application_train(df)
    num_feature_count = len(prep.artifacts.numeric_cols)

    print(f"Preprocessing complete.")
    print(f"X_train shape: {prep.X_train.shape} | X_valid shape: {prep.X_valid.shape}")
    print(f"Numeric cols: {num_feature_count} | One-hot cols: {prep.X_train.shape[1] - num_feature_count}")

    # 3) Train logistic regression
    trained_lr = train_logistic_regression(
        prep.X_train,
        prep.y_train,
        num_feature_count=num_feature_count,
        max_iter=1000,
    )
    print("Trained Logistic Regression.")

    # 4) Evaluate
    y_proba = predict_proba_logistic_regression(trained_lr, prep.X_valid)
    eval_result = evaluate_classifier(prep.y_valid, y_proba, threshold=args.threshold)

    print("Evaluation complete.")
    print(f"Threshold: {eval_result.threshold}")
    print(f"Precision: {eval_result.precision:.4f} | Recall: {eval_result.recall:.4f} | Accuracy: {eval_result.accuracy:.4f}")
    print(f"Confusion (tn fp / fn tp): {eval_result.tn} {eval_result.fp} / {eval_result.fn} {eval_result.tp}")
    print("Capture/Lift:")
    print(eval_result.capture_lift)

    # 5) Save outputs
    # Model bundle
    model_bundle = {
        "model": trained_lr.model,
        "scaler": trained_lr.scaler,
        "num_feature_count": trained_lr.num_feature_count,
        "feature_names": prep.artifacts.feature_names,
    }
    model_path = paths.models_dir / "logistic_regression.joblib"
    joblib.dump(model_bundle, model_path)

    # Preprocess artifacts (imputer + ohe + dropped columns + col lists)
    preprocess_path = paths.models_dir / "preprocess_artifacts.joblib"
    joblib.dump(prep.artifacts, preprocess_path)

    # Metrics JSON
    metrics_path = paths.reports_dir / "metrics.json"
    save_eval_result(eval_result, metrics_path)

    print(f"Saved model bundle: {model_path}")
    print(f"Saved preprocess artifacts: {preprocess_path}")
    print(f"Saved metrics: {metrics_path}")
    print("=== Pipeline completed successfully ===")


if __name__ == "__main__":
    main()