# Flask traffic prediction API

Inference backend for the Traffic Prediction app. The live prediction path uses the **Full Spatial GRU**. It does not retrain models, refit scalers, or invent historical traffic.

This is a **historical simulation**. The dataset ends at 2017-06-30 23:00. The API does not claim real-time prediction. Weather, occupancy, GPS, and accident/event inputs are **not** used.

The original 13-feature baseline GRU service remains in `backend/services/prediction_service.py` as a fallback/reference. History still reads the baseline processed series. `POST /api/predict` uses Spatial GRU only.

## Model

- Application model: **Spatial_GRU**
- Weights: `models/trained/gru/spatial/model.keras`
- Feature scaler: `models/artifacts/spatial/feature_scaler.joblib` (spatial train only)
- Target scaler: `models/artifacts/spatial/target_scaler.joblib`
- Feature config: `models/artifacts/spatial/feature_config.json`
- Input: 168 hourly rows, **53** leakage-safe features (temporal + own lags + lagged network/cross-junction traffic)
- Output: one-step-ahead `Vehicles` in original units
- Architecture: GRU(128, return_sequences=True) → Dropout(0.2) → GRU(64) → Dense(32, ReLU) → Dense(1, linear)

`Vehicles(T)` is never a model input. If an actual count exists at `T`, the API may return it for display after prediction.

## Limitation

A prediction timestamp `T` is served only when that junction has **168 consecutive hourly spatial feature rows ending at T** in `data/processed/spatial/`. Lags at `T` use traffic through `T-1` only. Four-junction spatial history is required; missing Junction 4 (or any required lag) is not filled with 0, mean, or interpolation.

These requests return HTTP 400:

- Dates after 2017-06-30 23:00 (including `2026-10-02T10:00:00`): historical dataset has no data for that timestamp
- Hours with no processed spatial row or with a shorter lookback: `Insufficient historical data for the requested spatial prediction timestamp.`

Valid example: `2017-06-15T10:00:00` for junctions 1–4.

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

Verify without leaving a server running:

```bash
$env:KERAS_BACKEND="torch"
$env:PYTHONUNBUFFERED=1
.\.venv\Scripts\python.exe -m backend.verify_spatial_inference
```

Base URL: `http://127.0.0.1:5000`

CORS is enabled for Vite ports 5173–5175.

Startup logs should include Spatial GRU and spatial scaler load messages. If loading fails, the process exits. There is no mock-prediction fallback.

## Endpoints

### GET /api/health

```json
{ "status": "ok", "model": "Spatial_GRU" }
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

`lat` / `lng` / `area` are UI display fields (not in Kaggle `traffic.csv` and not used by the model).

### GET /api/history

Required: `junction=1` or `locationId=j1`. Optional: `start`, `end` (hour-aligned). With no range, the last 48 hours for that junction are returned from the baseline processed history.

```bash
curl "http://127.0.0.1:5000/api/history?junction=1"
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

```json
{
  "locationId": "j1",
  "junction": 1,
  "datetime": "2017-06-15T10:00:00",
  "start": "2017-06-15T10:00:00",
  "end": "2017-06-15T10:00:00",
  "predicted_vehicles": 86.44,
  "model": "Spatial_GRU",
  "features": 53,
  "lookback": 168,
  "historical_simulation": true,
  "points": [
    {
      "timestamp": "2017-06-15T10:00:00",
      "predicted": 86.44,
      "actual": 90.0,
      "congestion": "severe"
    }
  ],
  "peakCongestion": "severe",
  "peakVolume": 86.44
}
```

`actual` is the recorded Vehicles value at `T` when present. It is not used as a model input.

## Errors

JSON `{ "error": "..." }` with no Python traceback.

| Status | When |
| --- | --- |
| 400 | Invalid junction, missing fields, invalid datetime, non-hourly timestamp, timestamp after dataset end, insufficient spatial history |
| 404 | Unknown path |
| 500 | Inference failure |
| 503 | Health check if the Spatial GRU is not loaded |
