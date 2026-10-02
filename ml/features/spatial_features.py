"""Leakage-safe historical spatial/network lags. Does not use Vehicles(T)."""

from __future__ import annotations

import pandas as pd

from ml.features.lag_features import LAG_HOURS, add_lag_features
from ml.features.temporal_features import add_temporal_features

GLOBAL_SPATIAL_COLUMNS = (
    "network_total_traffic",
    "network_avg_traffic",
    "junction_1_traffic",
    "junction_2_traffic",
    "junction_3_traffic",
    "junction_4_traffic",
)

ROW_SPATIAL_COLUMNS = (
    "other_junction_traffic",
    "other_junction_avg_traffic",
)

DROP_COLUMNS = ("ID", "latitude", "longitude", "gps_source")

TEMPORAL_COLUMNS = (
    "hour",
    "day_of_week",
    "day",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
)

OWN_LAG_COLUMNS = tuple(f"vehicles_lag_{lag}" for lag in LAG_HOURS)

GLOBAL_LAG_COLUMNS = tuple(
    f"{col}_lag_{lag}" for col in GLOBAL_SPATIAL_COLUMNS for lag in LAG_HOURS
)
ROW_LAG_COLUMNS = tuple(
    f"{col}_lag_{lag}" for col in ROW_SPATIAL_COLUMNS for lag in LAG_HOURS
)

SPATIAL_FEATURE_COLUMNS = [
    *TEMPORAL_COLUMNS,
    "Junction",
    *OWN_LAG_COLUMNS,
    *GLOBAL_LAG_COLUMNS,
    *ROW_LAG_COLUMNS,
]

TARGET_COLUMN = "Vehicles"
KEEP_COLUMNS = ["DateTime", *SPATIAL_FEATURE_COLUMNS, TARGET_COLUMN]


def drop_non_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    present = [col for col in DROP_COLUMNS if col in out.columns]
    if present:
        out = out.drop(columns=present)
    return out


def add_global_spatial_lags(
    df: pd.DataFrame,
    lags: tuple[int, ...] = LAG_HOURS,
) -> pd.DataFrame:
    """Attach snapshot values from T-lag. Missing timestamps stay NaN (not filled)."""
    missing = [col for col in GLOBAL_SPATIAL_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing spatial columns: {missing}")

    snap = (
        df.sort_values("DateTime")
        .drop_duplicates("DateTime", keep="first")[["DateTime", *GLOBAL_SPATIAL_COLUMNS]]
        .copy()
    )
    out = df.copy()
    for lag in lags:
        lagged = snap.copy()
        lagged["DateTime"] = lagged["DateTime"] + pd.Timedelta(hours=lag)
        lagged = lagged.rename(
            columns={col: f"{col}_lag_{lag}" for col in GLOBAL_SPATIAL_COLUMNS}
        )
        out = out.merge(lagged, on="DateTime", how="left")
    return out


def add_row_spatial_lags(
    df: pd.DataFrame,
    lags: tuple[int, ...] = LAG_HOURS,
) -> pd.DataFrame:
    """Lag target-aware other-junction columns within each Junction only."""
    missing = [col for col in ROW_SPATIAL_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing spatial columns: {missing}")

    out = df.sort_values(["Junction", "DateTime"]).copy()
    for col in ROW_SPATIAL_COLUMNS:
        grouped = out.groupby("Junction", sort=False)[col]
        for lag in lags:
            out[f"{col}_lag_{lag}"] = grouped.shift(lag)
    return out


def add_spatial_lag_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = LAG_HOURS,
) -> pd.DataFrame:
    out = add_global_spatial_lags(df, lags=lags)
    out = add_row_spatial_lags(out, lags=lags)
    return out


def build_spatial_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Temporal + own lags + spatial lags. Does not drop lag-NaN rows."""
    before = {int(j): int(n) for j, n in df.groupby("Junction").size().items()}
    out = drop_non_features(df)
    out["DateTime"] = pd.to_datetime(out["DateTime"])
    out["Junction"] = out["Junction"].astype(int)
    out = add_temporal_features(out)
    out = add_lag_features(out)
    out = add_spatial_lag_features(out)
    extra = [col for col in GLOBAL_SPATIAL_COLUMNS + ROW_SPATIAL_COLUMNS if col in out.columns]
    if extra:
        out = out.drop(columns=extra)
    keep = [col for col in KEEP_COLUMNS if col in out.columns]
    out = out.loc[:, keep]
    after = {int(j): int(n) for j, n in out.groupby("Junction").size().items()}
    stats = {
        "rows_before": sum(before.values()),
        "rows_after_feature_build": len(out),
        "rows_per_junction_before": before,
        "rows_per_junction_after_build": after,
        "features": list(SPATIAL_FEATURE_COLUMNS),
        "target": TARGET_COLUMN,
        "lags": list(LAG_HOURS),
        "note": (
            "Contemporaneous network/cross-junction columns at T are dropped. "
            "Only lagged values from T-1, T-2, T-3, T-24, T-168 are kept. "
            "Own-series vehicles_lag_* are retained. "
            "junction_J_traffic_lag_k duplicates vehicles_lag_k on rows for junction J."
        ),
    }
    return out.reset_index(drop=True), stats
