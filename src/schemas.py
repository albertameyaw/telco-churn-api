"""
Pydantic models describing the /predict request and response shapes.

Why this lives in src/ and not app/: these field names must exactly match
CATEGORICAL_FEATURES + NUMERIC_FEATURES in config.py (that's what the
pipeline was fit on), so it's a "domain" concern shared by training-side
sanity checks and the API, not something that belongs to the web layer
alone.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ChurnPredictionRequest(BaseModel):
    """
    One customer record, field-for-field matching the training data
    columns. Pydantic validates types AND, via Literal, restricts
    categorical fields to values the model actually saw during training -
    a typo like "Fmale" gets rejected with a clear 422 error instead of
    silently reaching the model as an unknown category.
    """

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100, description="Months as a customer")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class ChurnPredictionResponse(BaseModel):
    """
    churn: the class prediction as a human-readable label.
    churn_probability: the model's raw probability that this customer
    churns - useful because a retention team may want to act on
    "top 10% riskiest customers" rather than a fixed 0.5 cutoff.
    """

    churn: Literal["Yes", "No"]
    churn_probability: float = Field(ge=0, le=1)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
