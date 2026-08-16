"""
Central place for paths and column definitions.

Why this file exists: both train.py (fitting the pipeline) and app/main.py
(serving predictions) need to agree on exactly which columns are numeric vs
categorical. If that list lived in two places, an edit to one and not the
other would silently break predictions at serve time. One source of truth
avoids that class of bug entirely.
"""

from pathlib import Path

# Resolve paths relative to the project root, not the current working
# directory, so `python src/train.py` works the same whether you run it
# from the repo root or from inside src/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

# The raw Kaggle CSV's target column and its label values.
TARGET_COLUMN = "Churn"
POSITIVE_LABEL = "Yes"  # what "churn = true" looks like in the raw data

# customerID is a unique identifier, not a feature - it must be dropped
# before it ever reaches the model, or the model would "memorize" it.
ID_COLUMN = "customerID"

# Column groups drive the ColumnTransformer in pipeline.py.
# NUMERIC_FEATURES get scaled; CATEGORICAL_FEATURES get one-hot encoded.
NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42  # fixed seed so training results are reproducible
