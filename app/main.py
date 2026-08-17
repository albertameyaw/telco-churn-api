"""
FastAPI service exposing the trained churn model.

Run locally with:  uvicorn app.main:app --reload
"""

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException

from src.config import ALL_FEATURES, MODEL_PATH
from src.schemas import ChurnPredictionRequest, ChurnPredictionResponse, HealthResponse

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predicts whether a customer will churn.",
    version="1.0.0",
)

# Loaded once at process startup rather than per-request. If the artifact
# is missing, keep the app alive and report it via /health instead of
# crashing on import.
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness/readiness check, polled by Render and any orchestrator."""
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.post("/predict", response_model=ChurnPredictionResponse)
def predict(request: ChurnPredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training first to produce models/model.joblib.",
        )

    # Pipeline was fit on a DataFrame with these exact column names.
    input_df = pd.DataFrame([request.model_dump()])[ALL_FEATURES]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0, 1]

    return ChurnPredictionResponse(
        churn="Yes" if prediction == 1 else "No",
        churn_probability=float(probability),
    )
