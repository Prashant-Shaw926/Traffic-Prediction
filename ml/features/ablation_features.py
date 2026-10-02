"""Ablation feature groups. Subsets of leakage-safe lagged spatial columns."""

from __future__ import annotations

from ml.features.feature_pipeline import FEATURE_COLUMNS
from ml.features.lag_features import LAG_HOURS

BASELINE_FEATURE_COLUMNS = list(FEATURE_COLUMNS)

NETWORK_LAG_COLUMNS = [
    f"{col}_lag_{lag}"
    for col in ("network_total_traffic", "network_avg_traffic")
    for lag in LAG_HOURS
]

CROSS_JUNCTION_LAG_COLUMNS = [
    f"{col}_lag_{lag}"
    for col in (
        "junction_1_traffic",
        "junction_2_traffic",
        "junction_3_traffic",
        "junction_4_traffic",
        "other_junction_traffic",
        "other_junction_avg_traffic",
    )
    for lag in LAG_HOURS
]

NETWORK_FEATURE_COLUMNS = BASELINE_FEATURE_COLUMNS + NETWORK_LAG_COLUMNS
CROSS_JUNCTION_FEATURE_COLUMNS = BASELINE_FEATURE_COLUMNS + CROSS_JUNCTION_LAG_COLUMNS

EXPERIMENTS = {
    "network": {
        "name": "Network",
        "feature_group": "baseline_plus_network_lags",
        "features": NETWORK_FEATURE_COLUMNS,
        "n_features": len(NETWORK_FEATURE_COLUMNS),
    },
    "cross_junction": {
        "name": "Cross_Junction",
        "feature_group": "baseline_plus_cross_junction_lags",
        "features": CROSS_JUNCTION_FEATURE_COLUMNS,
        "n_features": len(CROSS_JUNCTION_FEATURE_COLUMNS),
    },
}

assert len(BASELINE_FEATURE_COLUMNS) == 13
assert len(NETWORK_FEATURE_COLUMNS) == 23
assert len(CROSS_JUNCTION_FEATURE_COLUMNS) == 43
assert len(NETWORK_LAG_COLUMNS) == 10
assert len(CROSS_JUNCTION_LAG_COLUMNS) == 30
