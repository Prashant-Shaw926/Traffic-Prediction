"""Flask backend settings. Inference uses the saved GRU only."""

from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("KERAS_BACKEND", "torch")

HOST = "127.0.0.1"
PORT = 5000
CORS_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5175",
)

MODEL_NAME = "GRU"
LOOKBACK = 168
HISTORY_DEFAULT_HOURS = 48

GRU_MODEL_PATH = REPO_ROOT / "models" / "trained" / "gru" / "full" / "model.keras"
FEATURE_SCALER_PATH = REPO_ROOT / "models" / "artifacts" / "feature_scaler.joblib"
TARGET_SCALER_PATH = REPO_ROOT / "models" / "artifacts" / "target_scaler.joblib"

INSUFFICIENT_HISTORY = (
    "Insufficient historical data for the requested prediction timestamp."
)

# Display metadata for the map. Kaggle traffic.csv has no GPS; these match the UI catalog.
LOCATIONS = (
    {
        "id": "j1",
        "junction": 1,
        "name": "Junction 1",
        "area": "Silk Board",
        "lat": 12.9177,
        "lng": 77.6238,
    },
    {
        "id": "j2",
        "junction": 2,
        "name": "Junction 2",
        "area": "HSR 27th Main",
        "lat": 12.9121,
        "lng": 77.6446,
    },
    {
        "id": "j3",
        "junction": 3,
        "name": "Junction 3",
        "area": "Koramangala 80 ft",
        "lat": 12.9352,
        "lng": 77.6245,
    },
    {
        "id": "j4",
        "junction": 4,
        "name": "Junction 4",
        "area": "Agara",
        "lat": 12.9246,
        "lng": 77.6489,
    },
)

LOCATION_BY_ID = {item["id"]: item for item in LOCATIONS}
LOCATION_BY_JUNCTION = {item["junction"]: item for item in LOCATIONS}
