"""Verify Spatial GRU Flask inference. Does not train or modify artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import pandas as pd

from backend.config import (
    AFTER_DATASET_END,
    FEATURE_SCALER_PATH,
    GRU_MODEL_PATH,
    INSUFFICIENT_SPATIAL_HISTORY,
    MODEL_NAME,
    REPO_ROOT,
    SPATIAL_FEATURE_CONFIG_PATH,
    SPATIAL_FEATURE_SCALER_PATH,
    SPATIAL_GRU_MODEL_PATH,
    SPATIAL_N_FEATURES,
    SPATIAL_PROCESSED_DIR,
    SPATIAL_TARGET_SCALER_PATH,
    TARGET_SCALER_PATH,
)
from ml.training.sequence_generator import load_processed_frame

OFFLINE_PREDS = REPO_ROOT / "results" / "predictions" / "spatial_gru_test_predictions.csv"
COMPARE_TOLERANCE = 1e-4
FROZEN_PATHS = (
    GRU_MODEL_PATH,
    FEATURE_SCALER_PATH,
    TARGET_SCALER_PATH,
    SPATIAL_GRU_MODEL_PATH,
    SPATIAL_FEATURE_SCALER_PATH,
    SPATIAL_TARGET_SCALER_PATH,
    SPATIAL_FEATURE_CONFIG_PATH,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _hash_existing(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(path): _sha256(path) for path in paths if path.is_file()}


def _fail(name: str, detail: str) -> None:
    raise SystemExit(f"FAIL  {name}: {detail}")


def _ok(name: str, detail: str) -> None:
    print(f"PASS  {name}: {detail}")


def _predict(client, payload: dict):
    return client.post("/api/predict", json=payload)


def main() -> int:
    hashes_before = _hash_existing(FROZEN_PATHS)
    spatial_frame = load_processed_frame(processed_dir=SPATIAL_PROCESSED_DIR)
    spatial_frame["DateTime"] = pd.to_datetime(spatial_frame["DateTime"])
    spatial_min = spatial_frame["DateTime"].min()
    spatial_max = spatial_frame["DateTime"].max()
    insufficient_ts = pd.Timestamp("2017-01-07T23:00:00")
    if insufficient_ts >= spatial_min:
        insufficient_ts = spatial_min - pd.Timedelta(hours=1)
    print(
        f"Processed spatial DateTime range: {spatial_min} -> {spatial_max}; "
        f"insufficient-history test timestamp={insufficient_ts}"
    )

    from backend.app import app

    client = app.test_client()

    health = client.get("/api/health")
    if health.status_code != 200:
        _fail("health", f"status={health.status_code} body={health.get_json()}")
    health_body = health.get_json()
    if health_body.get("model") != "Spatial_GRU":
        _fail("health_model", f"model={health_body.get('model')}")
    _ok("health", json.dumps(health_body))

    locations = client.get("/api/locations")
    if locations.status_code != 200:
        _fail("locations", f"status={locations.status_code}")
    loc_body = locations.get_json()
    if not loc_body.get("locations"):
        _fail("locations", "empty locations")
    _ok("locations", f"n={len(loc_body['locations'])}")

    history = client.get("/api/history?junction=1")
    if history.status_code != 200:
        _fail("history", f"status={history.status_code} body={history.get_json()}")
    hist_body = history.get_json()
    if not hist_body.get("history"):
        _fail("history", "empty history")
    _ok("history", f"rows={len(hist_body['history'])}")

    valid_examples = []
    for junction in (1, 2, 3, 4):
        response = _predict(client, {"junction": junction, "datetime": "2017-06-15T10:00:00"})
        body = response.get_json()
        if response.status_code != 200:
            _fail(f"predict_j{junction}", f"status={response.status_code} body={body}")
        pred = body.get("predicted_vehicles")
        if not isinstance(pred, (int, float)) or not math.isfinite(float(pred)):
            _fail(f"predict_j{junction}_numeric", f"predicted_vehicles={pred}")
        if body.get("model") != "Spatial_GRU":
            _fail(f"predict_j{junction}_model", f"model={body.get('model')}")
        if body.get("features") != SPATIAL_N_FEATURES or body.get("lookback") != 168:
            _fail(f"predict_j{junction}_meta", str(body))
        if body.get("historical_simulation") is not True:
            _fail(f"predict_j{junction}_sim", str(body.get("historical_simulation")))
        valid_examples.append((junction, float(pred)))
        _ok(
            f"predict_j{junction}",
            f"predicted_vehicles={float(pred):.4f} actual={body['points'][0].get('actual')}",
        )

    ranged = _predict(
        client,
        {
            "locationId": "j1",
            "start": "2017-06-15T10:00:00",
            "end": "2017-06-15T12:00:00",
        },
    )
    ranged_body = ranged.get_json()
    if ranged.status_code != 200:
        _fail("predict_range", f"status={ranged.status_code} body={ranged_body}")
    if len(ranged_body.get("points") or []) != 3:
        _fail("predict_range_points", f"points={ranged_body.get('points')}")
    _ok("predict_range", f"points={len(ranged_body['points'])}")

    invalid = _predict(client, {"junction": 9, "datetime": "2017-06-15T10:00:00"})
    if invalid.status_code != 400:
        _fail("invalid_junction", f"status={invalid.status_code} body={invalid.get_json()}")
    _ok("invalid_junction", str(invalid.get_json()))

    after = _predict(client, {"junction": 1, "datetime": "2017-07-01T10:00:00"})
    after_body = after.get_json()
    if after.status_code != 400 or after_body.get("error") != AFTER_DATASET_END:
        _fail("after_dataset_end", f"status={after.status_code} body={after_body}")
    _ok("after_dataset_end", after_body.get("error"))

    future = _predict(client, {"junction": 1, "datetime": "2026-10-02T10:00:00"})
    future_body = future.get_json()
    if future.status_code != 400 or future_body.get("error") != AFTER_DATASET_END:
        _fail("future_timestamp", f"status={future.status_code} body={future_body}")
    _ok("future_timestamp", future_body.get("error"))

    insufficient = _predict(
        client,
        {"junction": 1, "datetime": insufficient_ts.strftime("%Y-%m-%dT%H:%M:%S")},
    )
    insufficient_body = insufficient.get_json()
    if (
        insufficient.status_code != 400
        or insufficient_body.get("error") != INSUFFICIENT_SPATIAL_HISTORY
    ):
        _fail("insufficient_spatial_history", f"status={insufficient.status_code} body={insufficient_body}")
    _ok("insufficient_spatial_history", f"{insufficient_ts} -> {insufficient_body.get('error')}")

    offline = pd.read_csv(OFFLINE_PREDS, parse_dates=["DateTime"])
    target = pd.Timestamp("2017-06-15 10:00:00")
    match = offline.loc[
        (offline["DateTime"] == target) & (offline["Junction"].astype(int) == 1)
    ]
    if match.empty:
        _fail("offline_row", f"no offline row for {target} junction 1")
    offline_pred = float(match.iloc[0]["Predicted"])
    api_pred = valid_examples[0][1]
    abs_diff = abs(api_pred - offline_pred)
    print(
        f"offline_prediction={offline_pred:.6f} api_prediction={api_pred:.6f} "
        f"absolute_difference={abs_diff:.6g}"
    )
    if abs_diff >= COMPARE_TOLERANCE:
        _fail(
            "offline_vs_api",
            f"abs_diff={abs_diff:.6g} >= {COMPARE_TOLERANCE}; inspect feature order, scaler, window, timestamp, input",
        )
    _ok("offline_vs_api", f"abs_diff={abs_diff:.6g} < {COMPARE_TOLERANCE}")

    hashes_after = _hash_existing(FROZEN_PATHS)
    if hashes_before != hashes_after:
        _fail("frozen_hashes", "baseline/spatial model or scaler files changed")
    print("== SHA-256 (unchanged) ==")
    for path, digest in hashes_after.items():
        print(f"{digest}  {path}")
    _ok("frozen_hashes", "baseline and spatial model/scaler hashes unchanged")

    print("ALL CHECKS PASSED")
    print(f"model={MODEL_NAME} features={SPATIAL_N_FEATURES} lookback=168")
    print("No model was retrained. React was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
