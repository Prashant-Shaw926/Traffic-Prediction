"""Traffic prediction HTTP routes."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from backend.config import LOCATIONS, MODEL_NAME
from backend.services.history_service import history_request
from backend.services.spatial_prediction_service import (
    predict_request,
    spatial_model_loaded,
)
from backend.utils.responses import json_error

traffic_bp = Blueprint("traffic", __name__, url_prefix="/api")


@traffic_bp.get("/health")
def health():
    if not spatial_model_loaded():
        return json_error("Model is not loaded.", 503)
    return jsonify({"status": "ok", "model": MODEL_NAME})


@traffic_bp.get("/locations")
def locations():
    return jsonify({"locations": list(LOCATIONS)})


@traffic_bp.get("/history")
def history():
    payload = history_request(
        junction=request.args.get("junction"),
        location_id=request.args.get("locationId") or request.args.get("location_id"),
        start=request.args.get("start"),
        end=request.args.get("end"),
    )
    return jsonify(payload)


@traffic_bp.post("/predict")
def predict():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return json_error("Missing request fields: JSON body is required.", 400)
    return jsonify(predict_request(body))
