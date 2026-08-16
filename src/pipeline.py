"""
Builds the preprocessing + model pipeline.

Key idea (this is the part the assignment is really testing): a
sklearn.Pipeline bundles preprocessing and the model into ONE object.
When we joblib.dump() the fitted pipeline, we're saving the scaler's
learned mean/variance, the encoder's learned categories, AND the model's
learned weights, all together. At prediction time the API calls
pipeline.predict(raw_dataframe) and every transform happens exactly as
it did during training - there is no separate "preprocessing script" the
API has to remember to run first. That's what "no train/serve skew" means:
skew happens when training-time and serving-time preprocessing code
drift apart. Baking it into one artifact makes that class of bug
impossible.
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
    Two parallel mini-pipelines, one per feature type:

    - Numeric: fill missing values with the median, then scale to
      mean 0 / variance 1. Scaling matters a lot for Logistic Regression
      (it's distance/gradient based) and is harmless for XGBoost (tree
      splits don't care about feature scale) - so one preprocessor works
      for both models.
    - Categorical: fill missing values with the most frequent category,
      then one-hot encode. handle_unknown="ignore" means if the live API
      ever sees a category value that didn't appear during training
      (e.g. a new PaymentMethod added later), it encodes as all-zeros
      instead of crashing the request.
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
    Baseline model. Logistic Regression is simple, fast, and interpretable
    (each feature gets a signed weight) - it's the "does this problem even
    need something fancier" sanity check before reaching for XGBoost.

    class_weight="balanced" matters because churn datasets are imbalanced
    (far more "stayed" than "left" customers). Without it, the model could
    get high accuracy just by always predicting "no churn" - balanced
    weighting penalizes mistakes on the minority class more, so the model
    actually has to learn to spot churners.
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
    XGBoost: an ensemble of decision trees, each one correcting the
    previous trees' mistakes ("boosting"). It usually beats a single
    linear model on tabular data with mixed feature types because it can
    learn non-linear interactions (e.g. "month-to-month contract AND low
    tenure" is much riskier than either factor alone) that Logistic
    Regression can't represent on its own.

    scale_pos_weight is XGBoost's equivalent of class_weight="balanced" -
    it's set at train time in train.py once we know the actual class
    ratio in the training split.
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
