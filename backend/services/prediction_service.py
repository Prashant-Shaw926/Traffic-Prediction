"""Load GRU artifacts once and run one-step-ahead inference. Never refit scalers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backend.config import (
    BASELINE_MODEL_NAME,
    FEATURE_SCALER_PATH,
    GRU_MODEL_PATH,
    INSUFFICIENT_HISTORY,
    LOCATION_BY_ID,
    LOCATION_BY_JUNCTION,
    LOOKBACK,
    TARGET_SCALER_PATH,
)
from backend.utils.responses import ApiError, congestion_from_volume, isoformat
from ml.features.feature_pipeline import FEATURE_COLUMNS, TARGET_COLUMN
from ml.inference.predictor import predict_vehicles
from ml.training.sequence_generator import load_processed_frame, load_scalers

logger = logging.getLogger(__name__)


@dataclass
class Artifacts:
    model: Any
    feature_scaler: Any
    target_scaler: Any
    frame: pd.DataFrame


_artifacts: Artifacts | None = None


def artifacts() -> Artifacts:
    if _artifacts is None:
        raise ApiError("Model is not loaded.", 503)
    return _artifacts


def model_loaded() -> bool:
    return _artifacts is not None


def load_artifacts() -> Artifacts:
    """Load GRU, scalers, and processed history once. Fail loudly; no mock fallback."""
    global _artifacts
    import keras

    if not GRU_MODEL_PATH.is_file():
        raise FileNotFoundError(f"GRU model not found: {GRU_MODEL_PATH}")
    if not FEATURE_SCALER_PATH.is_file() or not TARGET_SCALER_PATH.is_file():
        raise FileNotFoundError("Feature or target scaler is missing.")

    model = keras.models.load_model(GRU_MODEL_PATH)
    logger.info("Loaded GRU model")
    feature_scaler, target_scaler = load_scalers()
    logger.info("Loaded feature scaler")
    logger.info("Loaded target scaler")
    frame = load_processed_frame()
    frame["DateTime"] = pd.to_datetime(frame["DateTime"]).dt.floor("h")
    _artifacts = Artifacts(
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        frame=frame,
    )
    return _artifacts


def location_id_for_junction(junction: int) -> str:
    item = LOCATION_BY_JUNCTION.get(junction)
    if item is None:
        raise ApiError("Invalid junction.", 400)
    return str(item["id"])


def parse_junction(payload: dict[str, Any] | None = None, *, junction=None, location_id=None) -> int:
    raw_junction = junction
    raw_location = location_id
    if payload:
        if raw_junction is None:
            raw_junction = payload.get("junction")
        if raw_location is None:
            raw_location = payload.get("locationId") or payload.get("location_id")

    if raw_location is not None and str(raw_location).strip() != "":
        key = str(raw_location).strip().lower()
        if key in LOCATION_BY_ID:
            return int(LOCATION_BY_ID[key]["junction"])
        if key.isdigit() and int(key) in LOCATION_BY_JUNCTION:
            return int(key)
        raise ApiError("Invalid junction.", 400)

    if raw_junction is None or raw_junction == "":
        raise ApiError("Missing request fields: junction or locationId is required.", 400)
    try:
        number = int(raw_junction)
    except (TypeError, ValueError) as exc:
        raise ApiError("Invalid junction.", 400) from exc
    if number not in LOCATION_BY_JUNCTION:
        raise ApiError("Invalid junction.", 400)
    return number


def parse_timestamp(value: Any, *, field: str) -> pd.Timestamp:
    if value is None or str(value).strip() == "":
        raise ApiError(f"Missing request fields: {field} is required.", 400)
    try:
        stamp = pd.to_datetime(value)
    except (TypeError, ValueError) as exc:
        raise ApiError("Invalid datetime.", 400) from exc
    if pd.isna(stamp):
        raise ApiError("Invalid datetime.", 400)
    if getattr(stamp, "tzinfo", None) is not None:
        stamp = stamp.tz_localize(None)
    stamp = pd.Timestamp(stamp).floor("s")
    if stamp.minute != 0 or stamp.second != 0 or stamp.microsecond != 0:
        raise ApiError("Datetime must be aligned to a full hour.", 400)
    return stamp


def _hourly_range(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    if end < start:
        raise ApiError("End must be on or after start.", 400)
    hours = list(pd.date_range(start=start, end=end, freq="h"))
    if not hours:
        raise ApiError(INSUFFICIENT_HISTORY, 400)
    return hours


def resolve_prediction_hours(payload: dict[str, Any]) -> tuple[pd.Timestamp, pd.Timestamp, list[pd.Timestamp]]:
    if "datetime" in payload and payload.get("datetime") not in (None, ""):
        stamp = parse_timestamp(payload.get("datetime"), field="datetime")
        return stamp, stamp, [stamp]

    start_raw = payload.get("start")
    end_raw = payload.get("end")
    if start_raw in (None, "") and end_raw in (None, ""):
        raise ApiError("Missing request fields: datetime or start/end is required.", 400)
    start = parse_timestamp(start_raw, field="start")
    end = parse_timestamp(end_raw if end_raw not in (None, "") else start_raw, field="end")
    return start, end, _hourly_range(start, end)


def _series_for_junction(frame: pd.DataFrame, junction: int) -> pd.DataFrame:
    series = frame.loc[frame["Junction"].astype(int) == junction].sort_values("DateTime")
    return series.reset_index(drop=True)


def _window_ending_at(series: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    times = pd.to_datetime(series["DateTime"])
    matches = np.where(times == target)[0]
    if len(matches) == 0:
        raise ApiError(INSUFFICIENT_HISTORY, 400)
    end_idx = int(matches[0])
    start_idx = end_idx - LOOKBACK + 1
    if start_idx < 0:
        raise ApiError(INSUFFICIENT_HISTORY, 400)
    window = series.iloc[start_idx : end_idx + 1]
    diffs = pd.to_datetime(window["DateTime"]).diff().dropna()
    if len(window) != LOOKBACK or not (diffs == pd.Timedelta(hours=1)).all():
        raise ApiError(INSUFFICIENT_HISTORY, 400)
    if pd.to_datetime(window["DateTime"].iloc[-1]) != target:
        raise ApiError(INSUFFICIENT_HISTORY, 400)
    return window


def _predict_windows(windows: list[pd.DataFrame]) -> np.ndarray:
    store = artifacts()
    stacked = pd.concat(
        [window.loc[:, FEATURE_COLUMNS] for window in windows],
        ignore_index=True,
    )
    scaled = np.asarray(store.feature_scaler.transform(stacked), dtype=np.float32)
    x = scaled.reshape(len(windows), LOOKBACK, len(FEATURE_COLUMNS))
    try:
        return predict_vehicles(store.model, x, store.target_scaler)
    except Exception:
        logger.exception("GRU inference failed")
        raise ApiError("Prediction failed.", 500) from None


def predict_request(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        raise ApiError("Missing request fields.", 400)
    junction = parse_junction(payload)
    start, end, hours = resolve_prediction_hours(payload)
    store = artifacts()
    series = _series_for_junction(store.frame, junction)
    windows = [_window_ending_at(series, stamp) for stamp in hours]
    predicted = _predict_windows(windows)

    points = []
    for stamp, window, value in zip(hours, windows, predicted):
        actual_raw = window.iloc[-1][TARGET_COLUMN]
        actual = None if pd.isna(actual_raw) else float(actual_raw)
        pred = float(value)
        points.append(
            {
                "timestamp": isoformat(stamp),
                "predicted": pred,
                "actual": actual,
                "congestion": congestion_from_volume(pred),
            }
        )

    peak_point = max(points, key=lambda item: item["predicted"])
    location_id = location_id_for_junction(junction)
    return {
        "locationId": location_id,
        "junction": junction,
        "datetime": points[0]["timestamp"],
        "start": isoformat(start),
        "end": isoformat(end),
        "predicted_vehicles": points[0]["predicted"],
        "model": BASELINE_MODEL_NAME,
        "points": points,
        "peakCongestion": peak_point["congestion"],
        "peakVolume": peak_point["predicted"],
    }
