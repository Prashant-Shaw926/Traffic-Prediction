"""Per-junction lag features. Never mix one junction's history into another."""

from __future__ import annotations

import pandas as pd

LAG_HOURS = (1, 2, 3, 24, 168)


def add_lag_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = LAG_HOURS,
) -> pd.DataFrame:
    """Shift Vehicles within each Junction only. Leading lags stay NaN (not 0)."""
    out = df.sort_values(["Junction", "DateTime"]).copy()
    grouped = out.groupby("Junction", sort=False)["Vehicles"]
    for lag in lags:
        out[f"vehicles_lag_{lag}"] = grouped.shift(lag)
    return out
