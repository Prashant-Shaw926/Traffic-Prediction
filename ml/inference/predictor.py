"""Inverse-scaled vehicle predictions. Does not refit scalers."""

from __future__ import annotations

from typing import Any

import numpy as np


def predict_vehicles(
    model: Any,
    X_scaled: np.ndarray,
    target_scaler: Any,
) -> np.ndarray:
    """Model outputs scaled targets; return original Vehicles units."""
    y_scaled = model.predict(X_scaled, verbose=0)
    return target_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()


def inverse_target(y_scaled: np.ndarray, target_scaler: Any) -> np.ndarray:
    return target_scaler.inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).ravel()
