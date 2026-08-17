"""
Builds the preprocessing + model pipeline.

The preprocessor and classifier are wrapped in a single sklearn.Pipeline
so joblib.dump() serializes both together. The API then calls
pipeline.predict() directly on raw request fields - there's no separate
preprocessing step to keep in sync between training and serving.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE


def build_preprocessor() -> ColumnTransformer:
    """
    Numeric features: median imputation + standard scaling. Scaling is
    required for Logistic Regression and harmless for XGBoost, so one
    preprocessor serves both models.

    Categorical features: most-frequent imputation + one-hot encoding.
    handle_unknown="ignore" so an unseen category at inference time
    (e.g. a new PaymentMethod) encodes as all-zeros instead of raising.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])


def build_logistic_regression_pipeline() -> Pipeline:
    """
    Baseline model. class_weight="balanced" compensates for the ~73/27
    class imbalance in the churn label, so the model isn't rewarded for
    just predicting the majority class.
    """
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def build_xgboost_pipeline() -> Pipeline:
    """
    scale_pos_weight (XGBoost's equivalent of class_weight="balanced")
    is set in train.py once the training split's actual class ratio is
    known.
    """
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )),
    ])
