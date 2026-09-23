"""Calendar features derived only from DateTime."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, calendar, weekend, and cyclic hour encodings. No hour one-hot."""
    out = df.copy()
    dt = pd.to_datetime(out["DateTime"])
    hour = dt.dt.hour.astype(int)
    out["hour"] = hour
    out["day_of_week"] = dt.dt.dayofweek.astype(int)
    out["day"] = dt.dt.day.astype(int)
    out["month"] = dt.dt.month.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out
