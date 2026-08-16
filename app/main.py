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

# Loaded once at process startup, not per-request. joblib.load deserializes
# the entire fitted Pipeline - preprocessor and model together - so a
# single object is enough to go from raw customer fields to a prediction.
# If the file is missing (e.g. someone forgot to run training before
# starting the API), we keep the app alive but report it via /health
# rather than crashing on import - that makes the failure diagnosable
# instead of just "the container won't start."
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


@app.get("/health", response_model=HealthResponse)
def health():
    """
    Liveness/readiness check. Render (and any orchestrator) polls this to
    know whether the container is ready to receive traffic. Reporting
    model_loaded separately from a 200 status lets you tell "the process
    is up" apart from "the process is up but has no model to serve."
    """
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.post("/predict", response_model=ChurnPredictionResponse)
def predict(request: ChurnPredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training first to produce models/model.joblib.",
        )

    # The pipeline was fit on a DataFrame with these exact column names,
    # so prediction input needs the same shape: a one-row DataFrame, not
    # a plain dict. This is where ColumnTransformer earns its keep - we
    # hand it the raw request fields untouched and it applies the exact
    # same imputation/scaling/encoding that training used.
    input_df = pd.DataFrame([request.model_dump()])[ALL_FEATURES]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0, 1]

    return ChurnPredictionResponse(
        churn="Yes" if prediction == 1 else "No",
        churn_probability=float(probability),
    )
