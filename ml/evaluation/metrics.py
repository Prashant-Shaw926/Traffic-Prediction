"""Regression metrics on original Vehicles units (after inverse scaling)."""

from __future__ import annotations

import numpy as np


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """RMSE, MAE, MAPE (%), R^2. MAPE uses max(|y|, epsilon) so zeros do not explode."""
    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    err = pred - true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    denom = np.maximum(np.abs(true), epsilon)
    mape = float(np.mean(np.abs(err) / denom) * 100.0)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    r2 = float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def format_metrics(metrics: dict[str, float]) -> str:
    r2 = metrics["R2"]
    r2_text = "nan" if np.isnan(r2) else f"{r2:.4f}"
    return (
        f"RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  "
        f"MAPE={metrics['MAPE']:.4f}%  R2={r2_text}"
    )
