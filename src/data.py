"""
Loading and cleaning the raw Telco Customer Churn CSV.

Kept separate from pipeline.py on purpose: this file deals with *data
hygiene* (fixing types, dropping unusable rows) which must happen once,
at training time, on the full historical dataset. pipeline.py deals with
*feature transforms* (scaling, encoding) which must happen identically
at both training time and prediction time. Mixing the two would make it
easy to accidentally apply a cleaning step at train time that the API
never replicates at serve time.
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
    Load the dataset and apply the fixes every downstream step relies on.

    Two known issues with this specific dataset:
    1. TotalCharges is read as a string because a handful of rows (new
       customers with tenure=0) have it as a blank " " instead of a
       number. pd.to_numeric with errors="coerce" turns those into NaN,
       which the pipeline's imputer will then fill in.
    2. Churn (the target) is the strings "Yes"/"No". Models need numbers,
       so we map it to 1/0 here rather than inside the pipeline, since
       the target column never goes through ColumnTransformer.
    """
    df = load_raw(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df[TARGET_COLUMN] = (df[TARGET_COLUMN] == POSITIVE_LABEL).astype(int)

    df = df.drop(columns=[ID_COLUMN])

    return df
