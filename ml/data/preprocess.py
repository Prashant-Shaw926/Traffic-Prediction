"""Run cleaning, features, temporal split, and train-only scaling. No model training."""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ml.data.clean_data import MAX_SHORT_GAP_HOURS, clean_traffic, format_outlier_report
from ml.data.load_data import REPO_ROOT, load_raw_traffic
from ml.features.feature_pipeline import FEATURE_COLUMNS, TARGET_COLUMN, build_features
from ml.features.lag_features import LAG_HOURS

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RESULTS_DIR = REPO_ROOT / "results"
ARTIFACTS_DIR = REPO_ROOT / "models" / "artifacts"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """70/15/15 split per Junction on chronological rows. No shuffle."""
    trains: list[pd.DataFrame] = []
    vals: list[pd.DataFrame] = []
    tests: list[pd.DataFrame] = []
    boundaries: dict[str, Any] = {}

    for junction, group in df.groupby("Junction", sort=True):
        group = group.sort_values("DateTime").reset_index(drop=True)
        n = len(group)
        n_train = int(n * TRAIN_FRAC)
        n_val = int(n * VAL_FRAC)
        train = group.iloc[:n_train].copy()
        val = group.iloc[n_train : n_train + n_val].copy()
        test = group.iloc[n_train + n_val :].copy()
        train["split"] = "train"
        val["split"] = "validation"
        test["split"] = "test"
        trains.append(train)
        vals.append(val)
        tests.append(test)
        boundaries[str(int(junction))] = {
            "n": n,
            "n_train": len(train),
            "n_validation": len(val),
            "n_test": len(test),
            "train": _span(train),
            "validation": _span(val),
            "test": _span(test),
        }

    train_df = pd.concat(trains, ignore_index=True)
    val_df = pd.concat(vals, ignore_index=True)
    test_df = pd.concat(tests, ignore_index=True)
    return train_df, val_df, test_df, boundaries


def _span(frame: pd.DataFrame) -> dict[str, str]:
    if frame.empty:
        return {"start": "", "end": ""}
    return {
        "start": str(frame["DateTime"].min()),
        "end": str(frame["DateTime"].max()),
    }


def fit_scalers(
    train: pd.DataFrame,
) -> tuple[MinMaxScaler, MinMaxScaler]:
    """Fit MinMax scalers on training rows only."""
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(train[FEATURE_COLUMNS])
    target_scaler = MinMaxScaler()
    target_scaler.fit(train[[TARGET_COLUMN]])
    return feature_scaler, target_scaler


def run_checks(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_scaler: MinMaxScaler,
    all_featured: pd.DataFrame,
) -> list[tuple[str, bool, str]]:
    combined = pd.concat([train, val, test], ignore_index=True)
    checks: list[tuple[str, bool, str]] = []

    nan_count = int(combined[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum().sum())
    checks.append(
        ("no_nan_in_model_features", nan_count == 0, f"nan_cells={nan_count}")
    )

    dups = int(combined.duplicated(subset=["Junction", "DateTime"]).sum())
    checks.append(
        ("no_duplicate_junction_datetime", dups == 0, f"duplicates={dups}")
    )

    sorted_ok = True
    for _, group in combined.groupby("Junction"):
        times = group["DateTime"]
        if not times.is_monotonic_increasing:
            sorted_ok = False
            break
    checks.append(("sorted_within_junction", sorted_ok, "Junction then DateTime"))

    has_id = "ID" in combined.columns
    checks.append(("no_id_column", not has_id, f"ID_present={has_id}"))

    future_ok = True
    order_notes: list[str] = []
    for junction in sorted(combined["Junction"].unique()):
        tr = train.loc[train["Junction"] == junction, "DateTime"]
        va = val.loc[val["Junction"] == junction, "DateTime"]
        te = test.loc[test["Junction"] == junction, "DateTime"]
        ok = bool(tr.max() < va.min() <= va.max() < te.min())
        if not ok:
            future_ok = False
        order_notes.append(
            f"J{int(junction)} train_end={tr.max()} val=[{va.min()}, {va.max()}] "
            f"test_start={te.min()} ok={ok}"
        )
    checks.append(
        (
            "temporal_order_per_junction",
            future_ok,
            "train max < val min < test min; " + " | ".join(order_notes),
        )
    )

    train_min = train[FEATURE_COLUMNS].min().to_numpy(dtype=float)
    train_max = train[FEATURE_COLUMNS].max().to_numpy(dtype=float)
    scaler_on_train = np.allclose(feature_scaler.data_min_, train_min) and np.allclose(
        feature_scaler.data_max_, train_max
    )
    checks.append(
        (
            "scaler_fitted_on_train_only",
            bool(scaler_on_train),
            "feature_scaler.data_min_/data_max_ match train feature min/max",
        )
    )

    lag_ok = True
    for _, group in all_featured.sort_values(["Junction", "DateTime"]).groupby("Junction"):
        vehicles = group[TARGET_COLUMN].to_numpy()
        lag1 = group["vehicles_lag_1"].to_numpy()
        if len(group) > 1 and not np.allclose(lag1[1:], vehicles[:-1]):
            lag_ok = False
            break
    checks.append(
        (
            "lags_within_junction",
            lag_ok,
            "vehicles_lag_1[t] equals Vehicles[t-1] inside each Junction",
        )
    )

    isolated = True
    for junction, group in combined.groupby("Junction"):
        if not (group["Junction"] == junction).all():
            isolated = False
    checks.append(
        ("junctions_isolated", isolated, "rows never mix Junction labels")
    )
    return checks


def format_checks(checks: list[tuple[str, bool, str]]) -> str:
    lines = ["Traffic prediction - preprocessing validation checks", ""]
    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        lines.append(f"{status}  {name}: {detail}")
    lines.append("")
    lines.append("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")
    return "\n".join(lines) + "\n"


def format_preprocessing_report(
    original_rows: int,
    clean_stats: dict[str, Any],
    feature_stats: dict[str, Any],
    boundaries: dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> str:
    outliers = clean_stats["outliers"]
    lines = [
        "Traffic prediction - preprocessing report",
        "",
        "== Counts ==",
        f"original_rows={original_rows}",
        f"invalid_timestamps_dropped={clean_stats['invalid_timestamps_dropped']}",
        f"duplicate_junction_datetime_dropped={clean_stats['duplicate_junction_datetime_dropped']}",
        f"hourly_slots_inserted={clean_stats['hourly_slots_inserted']}",
        f"short_gaps_filled={clean_stats['short_gaps_filled']}",
        f"long_gap_rows_dropped={clean_stats['long_gap_rows_dropped']}",
        f"nonphysical_negative_dropped={clean_stats['nonphysical_negative_dropped']}",
        f"rows_after_clean={clean_stats['rows_after_clean']}",
        f"rows_dropped_lag_nan={feature_stats['rows_dropped_lag_nan']}",
        f"final_rows={feature_stats['rows_after_lag_drop']}",
        f"train_rows={len(train)}  validation_rows={len(val)}  test_rows={len(test)}",
        "",
        "== Lag NaN drops per Junction ==",
    ]
    for junction, n in sorted(feature_stats["dropped_lag_nan_per_junction"].items()):
        lines.append(f"  Junction {junction}: dropped={n} remaining={feature_stats['rows_per_junction_after'].get(junction, 0)}")
    lines.append("")
    lines.append("== Features ==")
    lines.append(", ".join(FEATURE_COLUMNS))
    lines.append(f"target={TARGET_COLUMN}")
    lines.append("ID is not a feature and is not in processed files.")
    lines.append("")
    lines.append("== Lag leakage note ==")
    lines.append(
        "Lags are computed on the full per-junction history BEFORE the temporal split. "
        "A validation or test row may use vehicles_lag_* from earlier timestamps. "
        "That is valid at inference: those counts would already be known. "
        "Forbidden: using a future Vehicles value as a feature, shuffling, "
        "or fitting the scaler on validation/test."
    )
    lines.append("")
    lines.append("== Temporal split boundaries (per Junction) ==")
    for junction, info in boundaries.items():
        lines.append(f"Junction {junction}: n={info['n']}")
        lines.append(
            f"  Train      {info['train']['start']} -> {info['train']['end']}  "
            f"n={info['n_train']}"
        )
        lines.append(
            f"  Validation {info['validation']['start']} -> {info['validation']['end']}  "
            f"n={info['n_validation']}"
        )
        lines.append(
            f"  Test       {info['test']['start']} -> {info['test']['end']}  "
            f"n={info['n_test']}"
        )
    lines.append("")
    lines.append("== Missing values in processed features ==")
    combined = pd.concat([train, val, test], ignore_index=True)
    lines.append(str(int(combined[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum().sum())))
    lines.append("")
    lines.append("== Outliers ==")
    lines.append(
        f"IQR flagged (retained)={outliers['total_iqr_flagged']}  "
        f"Z-score flagged (retained)={outliers['total_zscore_flagged']}"
    )
    lines.append(outliers["action"])
    lines.append("")
    lines.append("== Scaler ==")
    lines.append("type=MinMaxScaler")
    lines.append("fitted_on=train only")
    lines.append(f"feature_scaler={ARTIFACTS_DIR / 'feature_scaler.joblib'}")
    lines.append(f"target_scaler={ARTIFACTS_DIR / 'target_scaler.joblib'}")
    lines.append(f"feature_config={ARTIFACTS_DIR / 'feature_config.json'}")
    lines.append("")
    lines.append(f"Processed CSVs are unscaled. max_short_gap_hours={MAX_SHORT_GAP_HOURS}.")
    return "\n".join(lines) + "\n"


def main() -> int:
    raw = load_raw_traffic()
    original_rows = len(raw)
    cleaned, clean_stats = clean_traffic(raw)
    featured, feature_stats = build_features(cleaned)
    train, val, test, boundaries = temporal_split(featured)
    feature_scaler, target_scaler = fit_scalers(train)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

    joblib.dump(feature_scaler, ARTIFACTS_DIR / "feature_scaler.joblib")
    joblib.dump(target_scaler, ARTIFACTS_DIR / "target_scaler.joblib")
    config = {
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "lags": list(LAG_HOURS),
        "frequency": "1h",
        "split_fractions": {
            "train": TRAIN_FRAC,
            "validation": VAL_FRAC,
            "test": TEST_FRAC,
        },
        "max_short_gap_hours": MAX_SHORT_GAP_HOURS,
        "scaler": "MinMaxScaler",
        "scaler_fitted_on": "train",
        "processed_values": "unscaled",
    }
    (ARTIFACTS_DIR / "feature_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    (RESULTS_DIR / "outlier_report.txt").write_text(
        format_outlier_report(clean_stats),
        encoding="utf-8",
    )
    (RESULTS_DIR / "preprocessing_report.txt").write_text(
        format_preprocessing_report(
            original_rows,
            clean_stats,
            feature_stats,
            boundaries,
            train,
            val,
            test,
        ),
        encoding="utf-8",
    )
    checks = run_checks(train, val, test, feature_scaler, featured)
    (RESULTS_DIR / "preprocessing_checks.txt").write_text(
        format_checks(checks),
        encoding="utf-8",
    )

    print((RESULTS_DIR / "preprocessing_report.txt").read_text(encoding="utf-8"))
    print((RESULTS_DIR / "preprocessing_checks.txt").read_text(encoding="utf-8"))
    print((RESULTS_DIR / "outlier_report.txt").read_text(encoding="utf-8"))
    return 0 if all(passed for _, passed, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
