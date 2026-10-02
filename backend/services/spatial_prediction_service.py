"""Full Spatial GRU inference. Never refit scalers. Vehicles(T) is not a model input."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backend.config import (
    AFTER_DATASET_END,
    DATASET_END,
    INSUFFICIENT_SPATIAL_HISTORY,
    LOOKBACK,
    MODEL_NAME,
    SPATIAL_FEATURE_CONFIG_PATH,
    SPATIAL_FEATURE_SCALER_PATH,
    SPATIAL_GRU_MODEL_PATH,
    SPATIAL_N_FEATURES,
    SPATIAL_PROCESSED_DIR,
    SPATIAL_TARGET_SCALER_PATH,
)
from backend.services.prediction_service import (
    location_id_for_junction,
    parse_junction,
    resolve_prediction_hours,
)
from backend.utils.responses import ApiError, congestion_from_volume, isoformat
from ml.inference.predictor import predict_vehicles
from ml.training.sequence_generator import load_processed_frame, load_scalers

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Vehicles"


@dataclass
class SpatialArtifacts:
    model: Any
    feature_scaler: Any
    target_scaler: Any
    frame: pd.DataFrame
    feature_columns: list[str]


_spatial_artifacts: SpatialArtifacts | None = None


def spatial_artifacts() -> SpatialArtifacts:
    if _spatial_artifacts is None:
        raise ApiError("Model is not loaded.", 503)
    return _spatial_artifacts


def spatial_model_loaded() -> bool:
    return _spatial_artifacts is not None


def _load_feature_config() -> dict[str, Any]:
    if not SPATIAL_FEATURE_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Spatial feature_config missing: {SPATIAL_FEATURE_CONFIG_PATH}")
    return json.loads(SPATIAL_FEATURE_CONFIG_PATH.read_text(encoding="utf-8"))


def _validate_feature_config(config: dict[str, Any], frame: pd.DataFrame) -> list[str]:
    features = list(config.get("features") or [])
    lookback = int(config.get("lookback", -1))
    if lookback != LOOKBACK:
        raise AssertionError(
            f"Spatial feature_config lookback must be {LOOKBACK}, got {lookback}"
        )
    if len(features) != SPATIAL_N_FEATURES:
        raise AssertionError(
            f"Spatial feature_config must list {SPATIAL_N_FEATURES} features, got {len(features)}"
        )
    if TARGET_COLUMN in features:
        raise AssertionError("Vehicles must not appear in spatial feature_config features")
    missing = [col for col in features if col not in frame.columns]
    if missing:
        raise AssertionError(
            "Spatial processed CSV missing feature_config columns: " + ", ".join(missing)
        )
    csv_feature_order = [col for col in frame.columns if col in set(features)]
    if csv_feature_order != features:
        raise AssertionError(
            "Spatial processed CSV feature order does not match feature_config.json"
        )
    return features


def load_spatial_artifacts() -> SpatialArtifacts:
    """Load Spatial GRU, spatial scalers, and processed spatial history once."""
    global _spatial_artifacts
    import keras

    if not SPATIAL_GRU_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Spatial GRU model not found: {SPATIAL_GRU_MODEL_PATH}")
    if not SPATIAL_FEATURE_SCALER_PATH.is_file() or not SPATIAL_TARGET_SCALER_PATH.is_file():
        raise FileNotFoundError("Spatial feature or target scaler is missing.")

    config = _load_feature_config()
    frame = load_processed_frame(processed_dir=SPATIAL_PROCESSED_DIR)
    frame["DateTime"] = pd.to_datetime(frame["DateTime"]).dt.floor("h")
    feature_columns = _validate_feature_config(config, frame)

    model = keras.models.load_model(SPATIAL_GRU_MODEL_PATH)
    logger.info("Loaded Spatial GRU model")
    feature_scaler, target_scaler = load_scalers(artifacts_dir=SPATIAL_FEATURE_SCALER_PATH.parent)
    logger.info("Loaded spatial feature scaler")
    logger.info("Loaded spatial target scaler")

    n_in = int(getattr(feature_scaler, "n_features_in_", len(feature_columns)))
    if n_in != SPATIAL_N_FEATURES:
        raise AssertionError(
            f"Spatial feature scaler n_features_in_ must be {SPATIAL_N_FEATURES}, got {n_in}"
        )

    _spatial_artifacts = SpatialArtifacts(
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        frame=frame,
        feature_columns=feature_columns,
    )
    return _spatial_artifacts


def _dataset_end() -> pd.Timestamp:
    return pd.Timestamp(DATASET_END)


def _series_for_junction(frame: pd.DataFrame, junction: int) -> pd.DataFrame:
    series = frame.loc[frame["Junction"].astype(int) == junction].sort_values("DateTime")
    return series.reset_index(drop=True)


def _window_ending_at(series: pd.DataFrame, target: pd.Timestamp) -> pd.DataFrame:
    times = pd.to_datetime(series["DateTime"])
    matches = np.where(times == target)[0]
    if len(matches) == 0:
        raise ApiError(INSUFFICIENT_SPATIAL_HISTORY, 400)
    end_idx = int(matches[0])
    start_idx = end_idx - LOOKBACK + 1
    if start_idx < 0:
        raise ApiError(INSUFFICIENT_SPATIAL_HISTORY, 400)
    window = series.iloc[start_idx : end_idx + 1]
    diffs = pd.to_datetime(window["DateTime"]).diff().dropna()
    if len(window) != LOOKBACK or not (diffs == pd.Timedelta(hours=1)).all():
        raise ApiError(INSUFFICIENT_SPATIAL_HISTORY, 400)
    if pd.to_datetime(window["DateTime"].iloc[-1]) != target:
        raise ApiError(INSUFFICIENT_SPATIAL_HISTORY, 400)
    return window


def _feature_tensor(windows: list[pd.DataFrame], feature_columns: list[str]) -> np.ndarray:
    if TARGET_COLUMN in feature_columns:
        raise AssertionError("Vehicles(T) must not be included in the Spatial GRU feature tensor")
    stacked = pd.concat(
        [window.loc[:, feature_columns] for window in windows],
        ignore_index=True,
    )
    if TARGET_COLUMN in stacked.columns:
        raise AssertionError("Vehicles leaked into the Spatial GRU feature matrix")
    if list(stacked.columns) != feature_columns:
        raise AssertionError("Spatial feature tensor column order does not match feature_config.json")
    scaled = np.asarray(
        spatial_artifacts().feature_scaler.transform(stacked),
        dtype=np.float32,
    )
    return scaled.reshape(len(windows), LOOKBACK, len(feature_columns))


def _predict_windows(windows: list[pd.DataFrame]) -> np.ndarray:
    store = spatial_artifacts()
    x = _feature_tensor(windows, store.feature_columns)
    try:
        return predict_vehicles(store.model, x, store.target_scaler)
    except Exception:
        logger.exception("Spatial GRU inference failed")
        raise ApiError("Prediction failed.", 500) from None


def _actual_at_target(window: pd.DataFrame) -> float | None:
    if TARGET_COLUMN not in window.columns:
        return None
    actual_raw = window.iloc[-1][TARGET_COLUMN]
    if pd.isna(actual_raw):
        return None
    return float(actual_raw)


def predict_request(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload:
        raise ApiError("Missing request fields.", 400)
    junction = parse_junction(payload)
    start, end, hours = resolve_prediction_hours(payload)
    dataset_end = _dataset_end()
    for stamp in hours:
        if stamp > dataset_end:
            raise ApiError(AFTER_DATASET_END, 400)

    store = spatial_artifacts()
    series = _series_for_junction(store.frame, junction)
    windows = [_window_ending_at(series, stamp) for stamp in hours]
    predicted = _predict_windows(windows)

    points = []
    for stamp, window, value in zip(hours, windows, predicted):
        pred = float(value)
        if not np.isfinite(pred):
            raise ApiError("Prediction failed.", 500)
        points.append(
            {
                "timestamp": isoformat(stamp),
                "predicted": pred,
                "actual": _actual_at_target(window),
                "congestion": congestion_from_volume(pred),
            }
        )

    peak_point = max(points, key=lambda item: item["predicted"])
    return {
        "locationId": location_id_for_junction(junction),
        "junction": junction,
        "datetime": points[0]["timestamp"],
        "start": isoformat(start),
        "end": isoformat(end),
        "predicted_vehicles": points[0]["predicted"],
        "model": MODEL_NAME,
        "features": SPATIAL_N_FEATURES,
        "lookback": LOOKBACK,
        "historical_simulation": True,
        "points": points,
        "peakCongestion": peak_point["congestion"],
        "peakVolume": peak_point["predicted"],
    }
