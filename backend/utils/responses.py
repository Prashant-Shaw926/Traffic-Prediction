"""JSON error helpers. Responses never include Python tracebacks."""

from __future__ import annotations

from typing import Any

from flask import jsonify


class ApiError(Exception):
    """HTTP-aware API error with a public JSON message."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def json_error(message: str, status_code: int):
    return jsonify({"error": message}), status_code


def congestion_from_volume(volume: float) -> str:
    """Same UI thresholds as src/data/mockTraffic.ts."""
    if volume < 26:
        return "clear"
    if volume < 51:
        return "moderate"
    if volume < 81:
        return "heavy"
    return "severe"


def isoformat(value: Any) -> str:
    stamp = getattr(value, "isoformat", None)
    if stamp is None:
        return str(value)
    text = stamp()
    if text.endswith("+00:00"):
        text = text[:-6]
    return text
