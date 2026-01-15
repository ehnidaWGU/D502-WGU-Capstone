from ingest import get_repo_root, get_raw_dir, validate_raw_files, load_application_train

def main() -> None:
    print("Starting ingestion pipeline...")

    # 1. Find repo root path
    project_root = get_repo_root()

    # 2. Locate raw data
    raw_dir = get_raw_dir()
    print(f"Using raw data directory: {raw_dir}")

    # 3. Validate raw files
    validate_raw_files(raw_dir)
    print("Raw file validation passed.")

    # 4. Load application_train.csv
    app_train_path = raw_dir / "application_train.csv"
    df = load_application_train(app_train_path)
    print(f"Loaded application_train.csv with shape: {df.shape}")

    # 5. Save to data/interim (CSV staging)
    interim_dir = project_root / "data" / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    out_path = interim_dir / "application_train.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved interim dataset to: {out_path}")
    print("Ingestion pipeline completed successfully.")

if __name__ == "__main__":
    main()