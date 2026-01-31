[![CI](https://github.com/ehnidaWGU/D502-WGU-Capstone/actions/workflows/ci.yml/badge.svg)](https://github.com/ehnidaWGU/D502-WGU-Capstone/actions/workflows/ci.yml)


# D502 WGU BSDA Capstone: Loan Default Prediction and Policy Simulation

My capstone for the WGU BSDA program. Utilizing the Home Credit dataset on Kaggle. This project builds a reproducible data cleaning pipeline, perform EDA, and develop a predictive model for calculating the odds of a loan defaulting as well as optimizing loan profits for the Home Credit bank. This project will additionally include an approval cutoff simulation and an interactive dashboard.

## Objectives:
- Build a reproducible data ingestion and preprocessing pipeline
- Build a baseline default risk model using application-level features
- Add engineered features from supporting tables via aggregation
- Evaluate model performance (capture and lift rate, precision and recall, feature ranking)
- Simulate approval cutoffs to illustrate risk–volume tradeoffs
- Provide a Tableau story for reporting and visualization

## Dataset
- Source: Kaggle - **Home Credit Default Risk**
- Raw files are stored locally and **will not** be committed to this GitHub Repo
- Processed artifacts, models, and metrics are generated programmatically


## Environment

- Language: Python 3.11+

- OS: Ubuntu (local development)

- Storage: SQLite

- Libraries: Pandas, NumPy, scikit-learn, joblib


## Quickstart

1. Clone the repository  
2. Create and activate a virtual environment, download the dataset here -> https://www.kaggle.com/competitions/home-credit-default-risk/data
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
4. run "python -m src.main"
5. run "python -m src.main --smoke-test"



## CI

GitHub Actions runs linting and tests on all pull requests and main

