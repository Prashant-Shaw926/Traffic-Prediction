"""Ablation preprocessing for network and cross-junction GRU experiments.

Reuses spatial overlap, lag construction, and 70/15/15. Does not write spatial or baseline artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ml.data.load_data import REPO_ROOT
from ml.data.preprocess_spatial import (
    TRAIN_FRAC,
    VAL_FRAC,
    TEST_FRAC,
    four_junction_overlap_mask,
    load_enriched,
    temporal_split,
    validate_enriched,
)
from ml.features.ablation_features import EXPERIMENTS
from ml.features.lag_features import LAG_HOURS
from ml.features.spatial_features import (
    KEEP_COLUMNS,
    SPATIAL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_spatial_features,
)

SPATIAL_KEYS_PATH = REPO_ROOT / "results" / "predictions" / "spatial_gru_comparison_ready.csv"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed" / "ablation"
ARTIFACTS_ROOT = REPO_ROOT / "models" / "artifacts" / "ablation"
RESULTS_DIR = REPO_ROOT / "results" / "ablation"

FROZEN_PATHS = (
    REPO_ROOT / "models" / "trained" / "gru" / "full" / "model.keras",
    REPO_ROOT / "models" / "artifacts" / "feature_scaler.joblib",
    REPO_ROOT / "models" / "artifacts" / "target_scaler.joblib",
    REPO_ROOT / "results" / "gru_metrics.csv",
    REPO_ROOT / "models" / "trained" / "gru" / "spatial" / "model.keras",
    REPO_ROOT / "models" / "artifacts" / "spatial" / "feature_scaler.joblib",
    REPO_ROOT / "models" / "artifacts" / "spatial" / "target_scaler.joblib",
    REPO_ROOT / "results" / "spatial_gru_metrics.csv",
    REPO_ROOT / "data" / "processed" / "spatial" / "test.csv",
    SPATIAL_KEYS_PATH,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _hash_existing(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(path): _file_sha256(path) for path in paths if path.is_file()}


def _key_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["DateTime", "Junction"]].copy()
    out["DateTime"] = pd.to_datetime(out["DateTime"])
    out["Junction"] = out["Junction"].astype(int)
    return out.sort_values(["Junction", "DateTime"]).reset_index(drop=True)


def fit_scalers(train: pd.DataFrame, feature_columns: list[str]) -> tuple[MinMaxScaler, MinMaxScaler]:
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(train[feature_columns])
    target_scaler = MinMaxScaler()
    target_scaler.fit(train[[TARGET_COLUMN]])
    return feature_scaler, target_scaler


def write_experiment(
    *,
    key: str,
    spec: dict[str, Any],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    spatial_keys: pd.DataFrame,
) -> dict[str, Any]:
    features: list[str] = list(spec["features"])
    keep = ["DateTime", *features, TARGET_COLUMN, "split"]
    train_out = train[keep].copy()
    val_out = val[keep].copy()
    test_out = test[keep].copy()

    test_keys = _key_frame(test_out)
    if not test_keys.equals(spatial_keys):
        raise AssertionError(
            f"{key}: ablation test keys do not match spatial comparison-ready keys "
            f"(n={len(test_keys)} vs {len(spatial_keys)})"
        )

    feature_scaler, target_scaler = fit_scalers(train_out, features)
    processed_dir = PROCESSED_ROOT / key
    artifacts_dir = ARTIFACTS_ROOT / key
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_out.to_csv(processed_dir / "train.csv", index=False)
    val_out.to_csv(processed_dir / "validation.csv", index=False)
    test_out.to_csv(processed_dir / "test.csv", index=False)
    joblib.dump(feature_scaler, artifacts_dir / "feature_scaler.joblib")
    joblib.dump(target_scaler, artifacts_dir / "target_scaler.joblib")
    config = {
        "experiment": key,
        "feature_group": spec["feature_group"],
        "features": features,
        "n_features": len(features),
        "target": TARGET_COLUMN,
        "lags": list(LAG_HOURS),
        "frequency": "1h",
        "split_fractions": {
            "train": TRAIN_FRAC,
            "validation": VAL_FRAC,
            "test": TEST_FRAC,
        },
        "scaler": "MinMaxScaler",
        "scaler_fitted_on": "ablation train only",
        "processed_values": "unscaled",
        "overlap": "timestamps with all four junctions observed",
        "lookback": 168,
        "gps_used": False,
        "contemporaneous_spatial_dropped": True,
    }
    (artifacts_dir / "feature_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    return {
        "experiment": key,
        "n_features": len(features),
        "train_rows": int(len(train_out)),
        "validation_rows": int(len(val_out)),
        "test_rows": int(len(test_out)),
        "test_start": str(test_out["DateTime"].min()),
        "test_end": str(test_out["DateTime"].max()),
        "processed_dir": str(processed_dir),
        "artifacts_dir": str(artifacts_dir),
    }


def main() -> int:
    hashes_before = _hash_existing(FROZEN_PATHS)
    spatial_keys = _key_frame(pd.read_csv(SPATIAL_KEYS_PATH, parse_dates=["DateTime"]))
    if len(spatial_keys) != 2508:
        raise AssertionError(f"Expected 2508 spatial keys, got {len(spatial_keys)}")

    raw = load_enriched()
    load_stats = validate_enriched(raw)
    if load_stats["invalid_timestamps"]:
        raw = raw.dropna(subset=["DateTime"])
    raw["Junction"] = raw["Junction"].astype(int)
    raw = raw.sort_values(["Junction", "DateTime"]).reset_index(drop=True)

    featured, _feature_stats = build_spatial_features(raw)
    overlap_mask = four_junction_overlap_mask(featured)
    overlapped = featured.loc[overlap_mask].copy()
    lag_na = overlapped[SPATIAL_FEATURE_COLUMNS].isna().any(axis=1)
    eligible = overlapped.loc[~lag_na, KEEP_COLUMNS].reset_index(drop=True)
    train, val, test, boundaries = temporal_split(eligible)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for key, spec in EXPERIMENTS.items():
        if len(spec["features"]) != spec["n_features"]:
            raise AssertionError(f"{key}: feature count mismatch")
        summaries.append(
            write_experiment(
                key=key,
                spec=spec,
                train=train,
                val=val,
                test=test,
                spatial_keys=spatial_keys,
            )
        )

    hashes_after = _hash_existing(FROZEN_PATHS)
    if hashes_before != hashes_after:
        raise AssertionError("Frozen baseline/spatial artifacts changed during ablation preprocessing")

    lines = [
        "Spatial ablation preprocessing",
        "",
        "Eligible rows match the spatial experiment (overlap + lag_168 drop on all spatial lags).",
        "Each ablation experiment keeps a feature subset and fits a new train-only MinMax scaler.",
        "Contemporaneous network/cross-junction values at T are not features.",
        "No 0/mean/interpolation fill. Baseline and full Spatial GRU artifacts were not written.",
        "",
        f"eligible_rows={len(eligible)} train={len(train)} validation={len(val)} test={len(test)}",
        f"spatial_test_keys={len(spatial_keys)}",
        f"test_start={test['DateTime'].min()} test_end={test['DateTime'].max()}",
        "",
    ]
    for junction, info in boundaries.items():
        span = info["test"]
        lines.append(
            f"Junction {junction} test {span['start']} -> {span['end']} n={info['n_test']}"
        )
    lines.append("")
    for item in summaries:
        lines.append(
            f"{item['experiment']}: n_features={item['n_features']} "
            f"test_rows={item['test_rows']} {item['test_start']} -> {item['test_end']}"
        )
        lines.append(f"  {item['processed_dir']}")
        lines.append(f"  {item['artifacts_dir']}")
    lines.append("")
    lines.append("Same-test-set keys: PASS")
    lines.append("Frozen artifacts unchanged: PASS")
    report = "\n".join(lines) + "\n"
    (RESULTS_DIR / "ablation_preprocessing_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print("Command: python -m ml.data.preprocess_ablation")
    print("No model training was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
