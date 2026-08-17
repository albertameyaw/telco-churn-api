"""
Loading and cleaning the raw Telco Customer Churn CSV.

Separate from pipeline.py: this handles one-off data hygiene applied at
training time, while pipeline.py handles feature transforms that must
run identically at both training and inference time.
"""

import pandas as pd

from src.config import DATA_PATH, ID_COLUMN, POSITIVE_LABEL, TARGET_COLUMN


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    """Read the CSV as-is, no cleaning."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download it from Kaggle (see README) and place it there."
        )
    return pd.read_csv(path)


def load_clean(path=DATA_PATH) -> pd.DataFrame:
    """
    Load the dataset and apply the fixes every downstream step relies on:
    TotalCharges arrives as a string (a handful of rows have a blank " "
    instead of a number) and gets coerced to numeric, with the resulting
    NaNs left for the pipeline's imputer to handle; and Churn gets mapped
    from "Yes"/"No" to 1/0 here since the target never passes through
    ColumnTransformer.
    """
    df = load_raw(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df[TARGET_COLUMN] = (df[TARGET_COLUMN] == POSITIVE_LABEL).astype(int)

    df = df.drop(columns=[ID_COLUMN])

    return df
