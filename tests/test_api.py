"""
Smoke tests for the API using FastAPI's TestClient (which wraps httpx and
doesn't need a running server - it calls the app in-process). These aren't
exhaustive; they exist to catch "the whole thing is broken" before you
deploy, e.g. a schema field that no longer matches the pipeline's expected
columns.

Run with:  pytest
Requires models/model.joblib to already exist (run `python -m src.train` first).
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_returns_valid_shape():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["churn"] in ("Yes", "No")
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_rejects_unknown_category():
    """
    Literal-typed fields should reject values outside the training
    vocabulary (e.g. a typo) with a 422, not silently pass it through.
    """
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["InternetService"] = "Satellite"
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    incomplete_payload = dict(VALID_PAYLOAD)
    del incomplete_payload["tenure"]
    response = client.post("/predict", json=incomplete_payload)
    assert response.status_code == 422
