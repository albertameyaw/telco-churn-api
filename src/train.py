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
    Accuracy alone is misleading on this ~73/27 imbalanced label - a
    model that always predicts "no churn" would score ~73% while being
    useless. Precision/recall/F1/ROC-AUC are reported alongside it to
    surface that failure mode.
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

    # stratify=y keeps the churn ratio consistent across train/test splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results = {}

    lr_pipeline = build_logistic_regression_pipeline()
    lr_pipeline.fit(X_train, y_train)
    results["logistic_regression"] = (lr_pipeline, evaluate(
        "Logistic Regression", lr_pipeline, X_test, y_test
    ))

    # majority/minority ratio, computed from the training split only to
    # avoid leaking test-set information into a fit-time parameter.
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_pipeline = build_xgboost_pipeline()
    xgb_pipeline.set_params(classifier__scale_pos_weight=scale_pos_weight)
    xgb_pipeline.fit(X_train, y_train)
    results["xgboost"] = (xgb_pipeline, evaluate(
        "XGBoost", xgb_pipeline, X_test, y_test
    ))

    # ROC-AUC is threshold-independent, so it's used as the selection
    # criterion rather than accuracy at the default 0.5 cutoff.
    best_name, (best_pipeline, best_metrics) = max(
        results.items(), key=lambda item: item[1][1]["roc_auc"]
    )

    print(f"\nBest model: {best_name} (roc_auc={best_metrics['roc_auc']:.4f})")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"Saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
