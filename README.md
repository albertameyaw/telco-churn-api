# Telco Customer Churn Prediction

A scikit-learn ML pipeline + FastAPI service that predicts whether a
telecom customer will churn, trained on the
[IBM Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Project layout

```
src/            Training-side code: config, data cleaning, pipeline, training script
app/main.py     FastAPI service (loads the trained pipeline, serves /health and /predict)
models/         Trained pipeline artifact (model.joblib)
data/           Raw CSV (not committed - see Setup)
tests/          API smoke tests
```

The preprocessing (imputation, scaling, one-hot encoding) is baked into
the saved `model.joblib` via a `sklearn.Pipeline` + `ColumnTransformer`.
The API never runs separate preprocessing code - it calls
`pipeline.predict()` on the raw request fields, so there's no risk of
training-time and serving-time preprocessing drifting apart.

## Setup

1. Python 3.11+ recommended (the Dockerfile uses 3.11 - that's the
   version with the most stable prebuilt wheels for these libraries).

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # macOS/Linux
   pip install -r requirements-dev.txt
   ```

3. Download the dataset and save it as `data/telco_churn.csv`:

   - Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
   - or the same data mirrored on GitHub:
     ```bash
     curl -L -o data/telco_churn.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
     ```

## Training

```bash
python -m src.train
```

This loads and cleans the data, trains Logistic Regression and XGBoost,
prints accuracy/precision/recall/F1/ROC-AUC for both, and saves whichever
scores higher on ROC-AUC to `models/model.joblib`.

## Running the API locally

```bash
uvicorn app.main:app --reload
```

Visit http://127.0.0.1:8000/docs for interactive Swagger docs (FastAPI
generates this automatically from the Pydantic schemas).

### Testing the API with curl

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "TotalCharges": 29.85
  }'
```

Expected response shape:

```json
{ "churn": "Yes", "churn_probability": 0.73 }
```

## Running tests

```bash
pytest
```

(Requires `models/model.joblib` to already exist - run training first.)

## Docker

Build and run locally:

```bash
docker build -t telco-churn-api .
docker run -p 8000:8000 telco-churn-api
```

Note the image bakes in whatever `models/model.joblib` exists on disk at
build time - retrain before rebuilding if you want an updated model.

## Deploying to Render

1. Push this repo to GitHub.
2. In the Render dashboard: **New > Web Service**, connect the repo.
3. Render auto-detects the `Dockerfile` - choose **Docker** as the
   environment (not "Python", which would look for a `requirements.txt`
   and expects you to specify a start command; the Dockerfile route is
   simpler here since it already defines the exact runtime).
4. Leave build/start commands blank - the Dockerfile's `CMD` handles it.
5. Instance type: the free/starter tier is enough for this model (it's
   small and CPU inference is fast).
6. Deploy. Render assigns a `$PORT` env var at runtime, which the
   Dockerfile's `CMD` already reads - no config needed on your end.
7. Once live, test with:
   ```bash
   curl https://<your-service>.onrender.com/health
   ```

See `MONITORING.md` for what to watch once this is serving real traffic.
