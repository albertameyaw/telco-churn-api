"""
Trains Logistic Regression and XGBoost on the Telco churn data, evaluates
both, and saves whichever pipeline scores higher on ROC-AUC as model.joblib.

Run with:  python -m src.train
"""

import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import ALL_FEATURES, MODEL_PATH, RANDOM_STATE, TARGET_COLUMN
from src.data import load_clean
from src.pipeline import build_logistic_regression_pipeline, build_xgboost_pipeline


def evaluate(name: str, pipeline, X_test, y_test) -> dict:
    """
    Computes all five required metrics. Why not just accuracy: churn is
    imbalanced (roughly 73% "no churn" / 27% "churn" in this dataset), so
    a model that always predicts "no churn" would score ~73% accuracy
    while being useless. Precision/recall/F1/ROC-AUC each surface that
    failure mode differently:
      - recall: of actual churners, how many did we catch? (misses here
        are the costly kind for a retention team)
      - precision: of everyone we flagged as a churn risk, how many
        really were? (misses here waste retention-offer budget)
      - F1: harmonic mean of the two, one number that punishes ignoring
        either
      - ROC-AUC: how well the model ranks churners above non-churners
        across ALL possible decision thresholds, not just 0.5 - useful
        because the "right" threshold is a business decision, not a
        modeling one
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n{name}")
    for metric_name, value in metrics.items():
        print(f"  {metric_name:>9}: {value:.4f}")

    return metrics


def main():
    df = load_clean()
    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    # stratify=y keeps the 73/27 class ratio consistent between train and
    # test splits - without it, an unlucky random split could leave the
    # test set with a noticeably different churn rate than training saw.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results = {}

    lr_pipeline = build_logistic_regression_pipeline()
    lr_pipeline.fit(X_train, y_train)
    results["logistic_regression"] = (lr_pipeline, evaluate(
        "Logistic Regression", lr_pipeline, X_test, y_test
    ))

    # scale_pos_weight tells XGBoost how much more to penalize missing a
    # churner vs. a false alarm. The standard formula is
    # (count of majority class) / (count of minority class), computed
    # from the TRAINING split only - using test data here would leak
    # information the model shouldn't have at fit time.
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_pipeline = build_xgboost_pipeline()
    xgb_pipeline.set_params(classifier__scale_pos_weight=scale_pos_weight)
    xgb_pipeline.fit(X_train, y_train)
    results["xgboost"] = (xgb_pipeline, evaluate(
        "XGBoost", xgb_pipeline, X_test, y_test
    ))

    # Pick the winner by ROC-AUC: it's threshold-independent, so it
    # reflects overall model quality rather than how lucky the default
    # 0.5 cutoff happened to be for one model over the other.
    best_name, (best_pipeline, best_metrics) = max(
        results.items(), key=lambda item: item[1][1]["roc_auc"]
    )

    print(f"\nBest model: {best_name} (roc_auc={best_metrics['roc_auc']:.4f})")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
