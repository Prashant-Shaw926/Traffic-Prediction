"""Compose temporal + lag features for the traffic volume target."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml.features.lag_features import LAG_HOURS, add_lag_features
from ml.features.temporal_features import add_temporal_features

FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "day",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "Junction",
    "vehicles_lag_1",
    "vehicles_lag_2",
    "vehicles_lag_3",
    "vehicles_lag_24",
    "vehicles_lag_168",
]

TARGET_COLUMN = "Vehicles"
LAG_COLUMNS = [f"vehicles_lag_{lag}" for lag in LAG_HOURS]
KEEP_COLUMNS = ["DateTime", *FEATURE_COLUMNS, TARGET_COLUMN]


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Engineer features, drop ID, drop rows that lack a full lag history."""
    before = {
        int(j): int(n) for j, n in df.groupby("Junction").size().items()
    }
    out = df.copy()
    if "ID" in out.columns:
        out = out.drop(columns=["ID"])
    out = add_temporal_features(out)
    out = add_lag_features(out)
    lag_na = out[LAG_COLUMNS].isna().any(axis=1)
    dropped = out.loc[lag_na]
    dropped_per_junction = {
        int(j): int(n) for j, n in dropped.groupby("Junction").size().items()
    }
    for junction in before:
        dropped_per_junction.setdefault(junction, 0)

    out = out.loc[~lag_na, KEEP_COLUMNS].reset_index(drop=True)
    after = {
        int(j): int(n) for j, n in out.groupby("Junction").size().items()
    }
    stats = {
        "rows_before_lag_drop": sum(before.values()),
        "rows_after_lag_drop": len(out),
        "rows_dropped_lag_nan": int(lag_na.sum()),
        "dropped_lag_nan_per_junction": dropped_per_junction,
        "rows_per_junction_after": after,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "note": (
            "Initial lag NaNs are dropped, not zero-filled. "
            "lag_24 is the prior 24 hours and lag_168 the prior 168 hours "
            "because the series is hourly and lags are computed per Junction."
        ),
    }
    return out, stats
