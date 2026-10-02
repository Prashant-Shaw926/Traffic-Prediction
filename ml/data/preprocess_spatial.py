"""Spatial/network preprocessing. Does not overwrite the baseline pipeline or train models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ml.data.load_data import REPO_ROOT
from ml.features.lag_features import LAG_HOURS
from ml.features.spatial_features import (
    GLOBAL_SPATIAL_COLUMNS,
    KEEP_COLUMNS,
    SPATIAL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_spatial_features,
)

ENRICHED_PATH = REPO_ROOT / "data" / "enriched" / "traffic_enriched_spatial_network.csv"
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "spatial"
RESULTS_DIR = REPO_ROOT / "results"
ARTIFACTS_DIR = REPO_ROOT / "models" / "artifacts" / "spatial"

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

BASELINE_HASH_PATHS = (
    REPO_ROOT / "results" / "lstm_metrics.csv",
    REPO_ROOT / "results" / "gru_metrics.csv",
    REPO_ROOT / "results" / "cnn_lstm_metrics.csv",
    REPO_ROOT / "data" / "processed" / "train.csv",
)


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_enriched() -> pd.DataFrame:
    if not ENRICHED_PATH.is_file():
        raise FileNotFoundError(f"Enriched dataset not found: {ENRICHED_PATH}")
    df = pd.read_csv(ENRICHED_PATH)
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    return df


def validate_enriched(df: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "original_rows": int(len(df)),
        "invalid_timestamps": int(df["DateTime"].isna().sum()),
        "duplicate_keys": int(df.duplicated(subset=["Junction", "DateTime"]).sum()),
        "negative_vehicles": int((pd.to_numeric(df["Vehicles"], errors="coerce") < 0).sum()),
        "junctions": sorted(int(j) for j in df["Junction"].dropna().unique()),
        "gps_non_null_lat": int(df["latitude"].notna().sum()) if "latitude" in df.columns else 0,
        "gps_non_null_lon": int(df["longitude"].notna().sum()) if "longitude" in df.columns else 0,
        "gps_source_values": sorted(df["gps_source"].dropna().astype(str).unique().tolist())
        if "gps_source" in df.columns
        else [],
        "sorted_within_junction": True,
        "coverage": {},
    }
    for junction, group in df.groupby("Junction", sort=True):
        times = group["DateTime"]
        if not times.is_monotonic_increasing:
            stats["sorted_within_junction"] = False
        stats["coverage"][str(int(junction))] = {
            "n": int(len(group)),
            "start": str(times.min()),
            "end": str(times.max()),
            "vehicles_nan": int(group["Vehicles"].isna().sum()),
        }
    return stats


def four_junction_overlap_mask(df: pd.DataFrame) -> pd.Series:
    counts = df.groupby("DateTime")["Junction"].nunique()
    overlap_times = counts[counts == 4].index
    return df["DateTime"].isin(overlap_times)


def temporal_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
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

    return (
        pd.concat(trains, ignore_index=True),
        pd.concat(vals, ignore_index=True),
        pd.concat(tests, ignore_index=True),
        boundaries,
    )


def _span(frame: pd.DataFrame) -> dict[str, str]:
    if frame.empty:
        return {"start": "", "end": ""}
    return {
        "start": str(frame["DateTime"].min()),
        "end": str(frame["DateTime"].max()),
    }


def fit_scalers(train: pd.DataFrame) -> tuple[MinMaxScaler, MinMaxScaler]:
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(train[SPATIAL_FEATURE_COLUMNS])
    target_scaler = MinMaxScaler()
    target_scaler.fit(train[[TARGET_COLUMN]])
    return feature_scaler, target_scaler


def _vehicles_lookup(source: pd.DataFrame) -> dict[tuple[pd.Timestamp, int], float]:
    lookup: dict[tuple[pd.Timestamp, int], float] = {}
    for row in source.itertuples(index=False):
        lookup[(pd.Timestamp(row.DateTime), int(row.Junction))] = float(row.Vehicles)
    return lookup


def run_checks(
    *,
    raw: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_scaler: MinMaxScaler,
    hashes_before: dict[str, str | None],
    hashes_after: dict[str, str | None],
) -> list[tuple[str, bool, str]]:
    combined = pd.concat([train, val, test], ignore_index=True)
    checks: list[tuple[str, bool, str]] = []

    dups = int(combined.duplicated(subset=["Junction", "DateTime"]).sum())
    checks.append(("no_duplicate_junction_datetime", dups == 0, f"duplicates={dups}"))

    sorted_ok = True
    for _, group in combined.groupby("Junction"):
        if not group["DateTime"].is_monotonic_increasing:
            sorted_ok = False
            break
    checks.append(("sorted_within_junction", sorted_ok, "DateTime increasing per Junction"))

    gps_cols = [col for col in ("latitude", "longitude", "gps_source") if col in combined.columns]
    checks.append(
        (
            "no_gps_in_processed_outputs",
            len(gps_cols) == 0,
            f"gps_columns={gps_cols}",
        )
    )
    lat_n = int(raw["latitude"].notna().sum()) if "latitude" in raw.columns else 0
    lon_n = int(raw["longitude"].notna().sum()) if "longitude" in raw.columns else 0
    checks.append(
        (
            "no_gps_values_fabricated",
            lat_n == 0 and lon_n == 0,
            f"source_non_null_lat={lat_n} lon={lon_n}",
        )
    )

    leaked_raw = [col for col in GLOBAL_SPATIAL_COLUMNS if col in SPATIAL_FEATURE_COLUMNS]
    checks.append(
        (
            "no_contemporaneous_spatial_features",
            len(leaked_raw) == 0 and TARGET_COLUMN not in SPATIAL_FEATURE_COLUMNS,
            f"forbidden_in_features={leaked_raw}",
        )
    )

    nan_count = int(combined[SPATIAL_FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum().sum())
    checks.append(("no_nan_in_model_features", nan_count == 0, f"nan_cells={nan_count}"))

    lookup = _vehicles_lookup(raw.dropna(subset=["DateTime"]))
    lag_ok = True
    checked = 0
    mismatches = 0
    sample = combined.sort_values(["Junction", "DateTime"])
    for rec in sample.itertuples(index=False):
        t = pd.Timestamp(rec.DateTime)
        prev = t - pd.Timedelta(hours=1)
        for junction in (1, 2, 3, 4):
            col = f"junction_{junction}_traffic_lag_1"
            expected = lookup.get((prev, junction))
            actual = getattr(rec, col)
            if expected is None or pd.isna(actual):
                continue
            checked += 1
            if not np.isclose(float(actual), float(expected), rtol=0, atol=1e-6):
                lag_ok = False
                mismatches += 1
                break
        if not lag_ok:
            break
    checks.append(
        (
            "no_spatial_feature_uses_vehicles_t",
            lag_ok and checked > 0,
            f"junction_k_traffic_lag_1 equals Vehicles of k at T-1; checked={checked} mismatches={mismatches}",
        )
    )

    future_ok = True
    order_notes: list[str] = []
    for junction in sorted(combined["Junction"].unique()):
        tr = train.loc[train["Junction"] == junction, "DateTime"]
        va = val.loc[val["Junction"] == junction, "DateTime"]
        te = test.loc[test["Junction"] == junction, "DateTime"]
        ok = bool(tr.max() < va.min() and va.max() < te.min())
        if not ok:
            future_ok = False
        order_notes.append(
            f"J{int(junction)} max_train={tr.max()} min_val={va.min()} "
            f"max_val={va.max()} min_test={te.min()} ok={ok}"
        )
    checks.append(
        (
            "datetime_split_boundaries_only",
            future_ok,
            "max(train DateTime) < min(val DateTime) and max(val DateTime) < min(test DateTime); "
            "Vehicles/feature overlap across splits is allowed. "
            + " | ".join(order_notes),
        )
    )

    train_min = train[SPATIAL_FEATURE_COLUMNS].min().to_numpy(dtype=float)
    train_max = train[SPATIAL_FEATURE_COLUMNS].max().to_numpy(dtype=float)
    scaler_on_train = np.allclose(feature_scaler.data_min_, train_min) and np.allclose(
        feature_scaler.data_max_, train_max
    )
    checks.append(
        (
            "scaler_fitted_on_train_only",
            bool(scaler_on_train),
            "spatial feature_scaler.data_min_/data_max_ match spatial train min/max",
        )
    )

    junction_ids = set(int(j) for j in combined["Junction"].unique())
    ids_ok = junction_ids == {1, 2, 3, 4}
    checks.append(("junction_identities_preserved", ids_ok, f"junctions={sorted(junction_ids)}"))

    baseline_ok = hashes_before == hashes_after
    checks.append(
        (
            "baseline_artifacts_unchanged",
            baseline_ok,
            "lstm/gru/cnn_lstm metrics and baseline train.csv hashes match",
        )
    )

    has_id = "ID" in combined.columns
    checks.append(("no_id_column", not has_id, f"ID_present={has_id}"))
    return checks


def format_checks(checks: list[tuple[str, bool, str]]) -> str:
    lines = [
        "Spatial/network preprocessing validation checks",
        "",
        "Split order is DateTime-only: max(train DateTime) < min(val DateTime)",
        "and max(val DateTime) < min(test DateTime). Overlapping Vehicles or",
        "feature values across splits are expected and are not a failure.",
        "",
    ]
    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        lines.append(f"{status}  {name}: {detail}")
    lines.append("")
    lines.append("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")
    return "\n".join(lines) + "\n"


def format_report(
    *,
    load_stats: dict[str, Any],
    feature_stats: dict[str, Any],
    overlap_hours: int,
    dropped_overlap: int,
    dropped_lag_nan: int,
    dropped_lag_nan_per_junction: dict[int, int],
    boundaries: dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> str:
    lines = [
        "Spatial/network preprocessing report",
        "",
        "This experiment is separate from the baseline LSTM/GRU/CNN-LSTM pipeline.",
        "No models were trained. Baseline processed CSVs and scalers were not written.",
        "",
        "== Counts ==",
        f"original_rows={load_stats['original_rows']}",
        f"invalid_timestamps={load_stats['invalid_timestamps']}",
        f"duplicate_keys={load_stats['duplicate_keys']}",
        f"rows_after_feature_build={feature_stats['rows_after_feature_build']}",
        f"four_junction_overlap_hours={overlap_hours}",
        f"rows_dropped_outside_overlap={dropped_overlap}",
        f"rows_dropped_lag_nan={dropped_lag_nan}",
        f"final_rows={len(train) + len(val) + len(test)}",
        f"train_rows={len(train)}  validation_rows={len(val)}  test_rows={len(test)}",
        f"n_features={len(SPATIAL_FEATURE_COLUMNS)}",
        "",
        "== Lag NaN drops per Junction (after overlap filter) ==",
    ]
    for junction, n in sorted(dropped_lag_nan_per_junction.items()):
        lines.append(f"  Junction {junction}: dropped={n}")
    lines.append("")
    lines.append("== Features ==")
    lines.append(", ".join(SPATIAL_FEATURE_COLUMNS))
    lines.append(f"target={TARGET_COLUMN}")
    lines.append("Excluded: ID, latitude, longitude, gps_source, unlagged spatial columns.")
    lines.append("")
    lines.append("== Missing-value handling ==")
    lines.append(
        "No zero/mean/interpolation fill for missing Junction 4 or spatial lags. "
        "Rows missing any required lag are dropped."
    )
    lines.append("")
    lines.append("== Junction 4 handling ==")
    lines.append(
        "Junction 4 begins 2017-01-01. The spatial experiment keeps only timestamps "
        "where all four junctions are observed, then drops rows until lag_168 exists "
        "for Junction 4 (expected first usable hour 2017-01-08 00:00). "
        "70/15/15 is applied on that eligible per-junction series, so calendar split "
        "dates differ from the 2015-2017 baseline. Pre-overlap Junction 1-3 hours are unused."
    )
    lines.append(
        "Network totals before Junction 4 existed are three-junction sums. Lags that "
        "look into that period keep those historical totals; Junction 4 is not backfilled."
    )
    j4 = load_stats["coverage"].get("4", {})
    lines.append(f"Junction 4 raw coverage: n={j4.get('n')} {j4.get('start')} -> {j4.get('end')}")
    lines.append("")
    lines.append("== Leakage ==")
    lines.append(
        "Features at T use traffic through T-1 only (own lags and spatial lags). "
        "Contemporaneous network_total_traffic / junction_*_traffic / other_junction_* "
        "at T are not model inputs. Cross-junction lags are simultaneous historical "
        "observations, not causal claims."
    )
    lines.append("")
    lines.append("Split validation uses DateTime boundaries only, not Vehicles values.")
    lines.append(feature_stats["note"])
    lines.append("")
    lines.append("== Temporal split (70/15/15 per Junction on eligible rows) ==")
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
    lines.append("== Scaler ==")
    lines.append("type=MinMaxScaler")
    lines.append("fitted_on=spatial train only")
    lines.append(f"feature_scaler={ARTIFACTS_DIR / 'feature_scaler.joblib'}")
    lines.append(f"target_scaler={ARTIFACTS_DIR / 'target_scaler.joblib'}")
    lines.append(f"feature_config={ARTIFACTS_DIR / 'feature_config.json'}")
    lines.append("Baseline models/artifacts/feature_scaler.joblib was not refit.")
    lines.append("")
    lines.append("Processed spatial CSVs store unscaled values.")
    return "\n".join(lines) + "\n"


def main() -> int:
    hashes_before = {str(path): _file_sha256(path) for path in BASELINE_HASH_PATHS}

    raw = load_enriched()
    load_stats = validate_enriched(raw)
    if load_stats["invalid_timestamps"]:
        raw = raw.dropna(subset=["DateTime"])
    raw["Junction"] = raw["Junction"].astype(int)
    raw = raw.sort_values(["Junction", "DateTime"]).reset_index(drop=True)

    featured, feature_stats = build_spatial_features(raw)

    overlap_mask = four_junction_overlap_mask(featured)
    overlap_hours = int(featured.loc[overlap_mask, "DateTime"].nunique())
    dropped_overlap = int((~overlap_mask).sum())
    overlapped = featured.loc[overlap_mask].copy()

    lag_na = overlapped[SPATIAL_FEATURE_COLUMNS].isna().any(axis=1)
    dropped_lag_nan = int(lag_na.sum())
    dropped_lag_nan_per_junction = {
        int(j): int(n)
        for j, n in overlapped.loc[lag_na].groupby("Junction").size().items()
    }
    for junction in (1, 2, 3, 4):
        dropped_lag_nan_per_junction.setdefault(junction, 0)
    eligible = overlapped.loc[~lag_na, KEEP_COLUMNS].reset_index(drop=True)

    train, val, test, boundaries = temporal_split(eligible)
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
        "experiment": "spatial_network",
        "features": list(SPATIAL_FEATURE_COLUMNS),
        "target": TARGET_COLUMN,
        "lags": list(LAG_HOURS),
        "frequency": "1h",
        "split_fractions": {
            "train": TRAIN_FRAC,
            "validation": VAL_FRAC,
            "test": TEST_FRAC,
        },
        "scaler": "MinMaxScaler",
        "scaler_fitted_on": "spatial train",
        "processed_values": "unscaled",
        "overlap": "timestamps with all four junctions observed",
        "lookback": 168,
        "gps_used": False,
    }
    (ARTIFACTS_DIR / "feature_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    hashes_after = {str(path): _file_sha256(path) for path in BASELINE_HASH_PATHS}
    checks = run_checks(
        raw=raw,
        train=train,
        val=val,
        test=test,
        feature_scaler=feature_scaler,
        hashes_before=hashes_before,
        hashes_after=hashes_after,
    )
    report = format_report(
        load_stats=load_stats,
        feature_stats=feature_stats,
        overlap_hours=overlap_hours,
        dropped_overlap=dropped_overlap,
        dropped_lag_nan=dropped_lag_nan,
        dropped_lag_nan_per_junction=dropped_lag_nan_per_junction,
        boundaries=boundaries,
        train=train,
        val=val,
        test=test,
    )
    (RESULTS_DIR / "spatial_preprocessing_report.txt").write_text(report, encoding="utf-8")
    (RESULTS_DIR / "spatial_preprocessing_checks.txt").write_text(
        format_checks(checks), encoding="utf-8"
    )

    print(report)
    print(format_checks(checks))
    print("Command: python -m ml.data.preprocess_spatial")
    print("No model training was run.")
    return 0 if all(passed for _, passed, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
