"""
Central place for paths and column definitions, shared by train.py and
app/main.py so preprocessing and serving can't drift out of sync.
"""

from pathlib import Path

# Relative to the project root rather than cwd, so this works whether
# invoked from the repo root or elsewhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "telco_churn.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

TARGET_COLUMN = "Churn"
POSITIVE_LABEL = "Yes"

# Unique identifier, not a feature - dropped before reaching the model.
ID_COLUMN = "customerID"

# Drives the ColumnTransformer in pipeline.py.
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
