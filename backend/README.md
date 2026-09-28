# Flask traffic prediction API

Inference backend for the Traffic Prediction app. It loads the **existing trained GRU** and the **existing train-only scalers**. It does not retrain models, refit scalers, or invent historical traffic.

React is not wired to this API yet.

## Model

- Weights: `models/trained/gru/full/model.keras`
- Feature scaler: `models/artifacts/feature_scaler.joblib`
- Target scaler: `models/artifacts/target_scaler.joblib`
- Input: 168 hourly rows, 13 features (same as training)
- Output: one-step-ahead `Vehicles` in original units

GRU test metrics (from `results/gru_metrics.csv`): RMSE 5.101930, MAE 3.207288, MAPE 13.479096%, R2 0.966585.

## Limitation

The Kaggle series ends at **2017-06-30 23:00**. A prediction timestamp `T` is served only when that junction has **168 consecutive hourly feature rows ending at T** in the processed CSVs. Lags at `T` use traffic through `T-1` only.

These requests return HTTP 400 (`Insufficient historical data for the requested prediction timestamp.`):

- Dates after 2017-06-30 23:00 (including `2017-07-01T10:00:00`)
- The React mock default window (today / 2026)
- Hours with no processed row or with a shorter lookback

Valid example: `2017-06-15T10:00:00` for junctions 1-4.

## Setup

Use the project virtualenv (Keras 3 + PyTorch backend). From the repository root:

```bash
py -3 -m venv .venv
.venv\Scripts\activate
pip install -r ml/requirements.txt
pip install -r backend/requirements.txt
```

`backend/requirements.txt` is Flask and Flask-CORS only. Do not install TensorFlow.

## Start

From the repository root:

```bash
$env:KERAS_BACKEND="torch"
$env:PYTHONUNBUFFERED=1
.\.venv\Scripts\python.exe -m flask --app backend.app run --host 127.0.0.1 --port 5000
```

Or:

```bash
.\.venv\Scripts\python.exe backend\app.py
```

Base URL: `http://127.0.0.1:5000`

CORS is enabled for `http://localhost:5173` and `http://127.0.0.1:5173` only.

Startup logs should include:

```
Loaded GRU model
Loaded feature scaler
Loaded target scaler
```

If loading fails, the process exits. There is no mock-prediction fallback.

## Endpoints

### GET /api/health

```json
{ "status": "ok", "model": "GRU" }
```

### GET /api/locations

```json
{
  "locations": [
    {
      "id": "j1",
      "junction": 1,
      "name": "Junction 1",
      "area": "Silk Board",
      "lat": 12.9177,
      "lng": 77.6238
    }
  ]
}
```

`lat` / `lng` / `area` are UI display fields (not in Kaggle `traffic.csv`).

### GET /api/history

Required: `junction=1` or `locationId=j1`. Optional: `start`, `end` (hour-aligned). With no range, the last 48 hours for that junction are returned.

```bash
curl "http://127.0.0.1:5000/api/history?junction=1"
```

```json
{
  "junction": 1,
  "locationId": "j1",
  "history": [
    {
      "datetime": "2017-06-29T00:00:00",
      "timestamp": "2017-06-29T00:00:00",
      "vehicles": 42.0,
      "actual": 42.0,
      "congestion": "moderate"
    }
  ]
}
```

### POST /api/predict

JSON body (frontend-shaped range, hourly points in `[start, end]`):

```json
{
  "locationId": "j1",
  "start": "2017-06-15T10:00:00",
  "end": "2017-06-15T12:00:00"
}
```

Single hour:

```json
{
  "junction": 1,
  "datetime": "2017-06-15T10:00:00"
}
```

```bash
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" -d "{\"locationId\":\"j1\",\"datetime\":\"2017-06-15T10:00:00\"}"
```

```json
{
  "locationId": "j1",
  "junction": 1,
  "datetime": "2017-06-15T10:00:00",
  "start": "2017-06-15T10:00:00",
  "end": "2017-06-15T10:00:00",
  "predicted_vehicles": 52.31,
  "model": "GRU",
  "points": [
    {
      "timestamp": "2017-06-15T10:00:00",
      "predicted": 52.31,
      "actual": 48.0,
      "congestion": "heavy"
    }
  ],
  "peakCongestion": "heavy",
  "peakVolume": 52.31
}
```

`actual` is the recorded Vehicles value at `T` when present. It is not used as a model input.

## Errors

JSON `{ "error": "..." }` with no Python traceback.

| Status | When |
| --- | --- |
| 400 | Invalid junction, missing fields, invalid datetime, non-hourly timestamp, insufficient history |
| 404 | Unknown path |
| 500 | Inference failure |
| 503 | Health check if the model is not loaded |
