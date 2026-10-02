"""Same-period evaluation of existing baseline GRU vs Spatial GRU. Does not train."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pandas as pd

from ml.data.load_data import REPO_ROOT
from ml.evaluation.metrics import format_metrics, regression_metrics
from ml.inference.predictor import inverse_target, predict_vehicles
from ml.training.config import LOOKBACK, N_FEATURES
from ml.training.sequence_generator import ARTIFACTS_DIR, build_sequences, load_scalers

SPATIAL_N_FEATURES = 53
SPATIAL_KEYS_PATH = REPO_ROOT / "results" / "predictions" / "spatial_gru_comparison_ready.csv"
BASELINE_MODEL_PATH = REPO_ROOT / "models" / "trained" / "gru" / "full" / "model.keras"
SPATIAL_MODEL_PATH = REPO_ROOT / "models" / "trained" / "gru" / "spatial" / "model.keras"
BASELINE_FEATURE_SCALER = ARTIFACTS_DIR / "feature_scaler.joblib"
BASELINE_TARGET_SCALER = ARTIFACTS_DIR / "target_scaler.joblib"
SPATIAL_FEATURE_SCALER = REPO_ROOT / "models" / "artifacts" / "spatial" / "feature_scaler.joblib"
SPATIAL_TARGET_SCALER = REPO_ROOT / "models" / "artifacts" / "spatial" / "target_scaler.joblib"

FROZEN_RESULT_PATHS = (
    REPO_ROOT / "results" / "gru_metrics.csv",
    REPO_ROOT / "results" / "gru_metrics.json",
    REPO_ROOT / "results" / "predictions" / "gru_test_predictions.csv",
    REPO_ROOT / "results" / "spatial_gru_metrics.csv",
    REPO_ROOT / "results" / "spatial_gru_metrics.json",
    REPO_ROOT / "results" / "predictions" / "spatial_gru_test_predictions.csv",
    SPATIAL_KEYS_PATH,
    REPO_ROOT / "results" / "model_comparison.csv",
    REPO_ROOT / "results" / "model_comparison.json",
)

PROTECTED_MODEL_PATHS = (
    BASELINE_MODEL_PATH,
    BASELINE_FEATURE_SCALER,
    BASELINE_TARGET_SCALER,
    SPATIAL_MODEL_PATH,
    SPATIAL_FEATURE_SCALER,
    SPATIAL_TARGET_SCALER,
)

OUT_BASELINE_PREDS = (
    REPO_ROOT / "results" / "predictions" / "baseline_gru_spatial_period_predictions.csv"
)
OUT_CSV = REPO_ROOT / "results" / "spatial_vs_baseline_gru_same_period.csv"
OUT_JSON = REPO_ROOT / "results" / "spatial_vs_baseline_gru_same_period.json"
OUT_TXT = REPO_ROOT / "results" / "spatial_vs_baseline_gru_same_period.txt"

METRIC_KEYS = ("RMSE", "MAE", "MAPE", "R2")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _hash_existing(paths: tuple[Path, ...]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths:
        if path.is_file():
            hashes[str(path)] = _file_sha256(path)
    return hashes


def _pct_change(spatial: float, baseline: float) -> float:
    denom = abs(baseline)
    if denom == 0:
        return float("nan")
    return (spatial - baseline) / denom * 100.0


def _metrics_by_junction(
    actual: np.ndarray,
    predicted: np.ndarray,
    junctions: np.ndarray,
) -> dict[str, dict[str, float]]:
    out = {"Overall": regression_metrics(actual, predicted)}
    for junction in (1, 2, 3, 4):
        mask = junctions == junction
        out[str(junction)] = regression_metrics(actual[mask], predicted[mask])
    return out


def _write_overall_csv(
    rows: list[dict[str, Any]],
    path: Path,
) -> None:
    fieldnames = ["Model", "RMSE", "MAE", "MAPE", "R2", "Test_Start", "Test_End", "Test_Rows"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Model": row["Model"],
                    "RMSE": f"{row['RMSE']:.6f}",
                    "MAE": f"{row['MAE']:.6f}",
                    "MAPE": f"{row['MAPE']:.6f}",
                    "R2": f"{row['R2']:.6f}",
                    "Test_Start": row["Test_Start"],
                    "Test_End": row["Test_End"],
                    "Test_Rows": row["Test_Rows"],
                }
            )


def _write_report(
    path: Path,
    *,
    test_start: str,
    test_end: str,
    n_rows: int,
    baseline_shape: tuple[int, ...],
    spatial_shape: tuple[int, ...],
    baseline_metrics: dict[str, dict[str, float]],
    spatial_metrics: dict[str, dict[str, float]],
    checks: list[tuple[str, bool, str]],
) -> None:
    lines = [
        "Same-test-period comparison: Baseline GRU vs Spatial GRU",
        "",
        "This is an evaluation-only experiment. Neither model was retrained.",
        "No overall ranking or composite score is assigned.",
        "",
        "== Evaluation period ==",
        f"Test_Start={test_start}",
        f"Test_End={test_end}",
        f"Test_Rows={n_rows}",
        "",
        "== Feature counts ==",
        f"Baseline feature count={N_FEATURES}",
        f"Spatial feature count={SPATIAL_N_FEATURES}",
        f"lookback={LOOKBACK}",
        f"Baseline matched input shape={baseline_shape}",
        f"Spatial test input shape={spatial_shape} (from Spatial GRU training; not regenerated)",
        "",
        "== Same-test-set verification ==",
    ]
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        lines.append(f"{status}  {name}: {detail}")
    all_pass = all(passed for _, passed, _ in checks)
    lines.append("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")

    lines.extend(
        [
            "",
            "== Overall metrics (original Vehicles) ==",
            f"Baseline_GRU  {format_metrics(baseline_metrics['Overall'])}",
            f"Spatial_GRU   {format_metrics(spatial_metrics['Overall'])}",
            "",
            "== Per-junction metrics ==",
        ]
    )
    for junction in ("1", "2", "3", "4"):
        lines.append(f"Junction {junction}")
        lines.append(f"  Baseline_GRU  {format_metrics(baseline_metrics[junction])}")
        lines.append(f"  Spatial_GRU   {format_metrics(spatial_metrics[junction])}")

    lines.extend(["", "== Absolute metric differences (Spatial - Baseline) =="])
    overall_diffs: dict[str, float] = {}
    overall_pcts: dict[str, float] = {}
    for key in METRIC_KEYS:
        diff = spatial_metrics["Overall"][key] - baseline_metrics["Overall"][key]
        pct = _pct_change(spatial_metrics["Overall"][key], baseline_metrics["Overall"][key])
        overall_diffs[key] = diff
        overall_pcts[key] = pct
        lines.append(f"{key}_diff={diff:.6f}")

    lines.extend(["", "== Percentage change ((Spatial - Baseline) / |Baseline| * 100) =="])
    for key in METRIC_KEYS:
        lines.append(f"{key}_pct_change={overall_pcts[key]:.4f}%")

    b = baseline_metrics["Overall"]
    s = spatial_metrics["Overall"]
    lines.extend(
        [
            "",
            "== Factual observations ==",
            (
                f"Spatial GRU RMSE was {s['RMSE']:.4f} compared with baseline GRU RMSE "
                f"{b['RMSE']:.4f} on the same test set."
            ),
            (
                f"Spatial GRU MAE was {s['MAE']:.4f} compared with baseline GRU MAE "
                f"{b['MAE']:.4f}."
            ),
            (
                f"Spatial GRU MAPE was {s['MAPE']:.4f}% compared with baseline GRU MAPE "
                f"{b['MAPE']:.4f}%."
            ),
            (
                f"Spatial GRU R2 was {s['R2']:.4f} compared with baseline GRU R2 "
                f"{b['R2']:.4f}."
            ),
            "These statements are not a ranking. Lower RMSE/MAE/MAPE and higher R2 are better individually.",
            "Existing results/model_comparison.csv was not updated.",
            "",
            "Command: python -m ml.evaluation.compare_spatial_vs_baseline_gru",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_checks(
    *,
    spatial: pd.DataFrame,
    baseline_preds: pd.DataFrame,
    actual_match_ok: bool,
    actual_max_diff: float,
    hashes_before: dict[str, str],
    hashes_after: dict[str, str],
    used_fit: bool,
    loaded_spatial_model: bool,
    baseline_shape: tuple[int, ...],
    n_features: int,
) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    spatial_keys = list(zip(spatial["DateTime"].astype(str), spatial["Junction"].astype(int)))
    baseline_keys = list(
        zip(baseline_preds["DateTime"].astype(str), baseline_preds["Junction"].astype(int))
    )
    checks.append(
        (
            "identical_datetime_junction_keys",
            spatial_keys == baseline_keys,
            f"spatial_rows={len(spatial_keys)} baseline_rows={len(baseline_keys)}",
        )
    )
    actual_equal = np.allclose(
        spatial["Actual"].to_numpy(dtype=float),
        baseline_preds["Actual"].to_numpy(dtype=float),
        rtol=0,
        atol=1e-6,
    )
    checks.append(
        (
            "identical_actual_values",
            actual_equal,
            "comparison-ready Actual reused as the shared target",
        )
    )
    checks.append(
        (
            "actual_matches_baseline_vehicles",
            actual_match_ok,
            f"max_abs_diff_vs_baseline_inverse_y={actual_max_diff:.6g}",
        )
    )
    dup_spatial = int(spatial.duplicated(["DateTime", "Junction"]).sum())
    dup_baseline = int(baseline_preds.duplicated(["DateTime", "Junction"]).sum())
    checks.append(
        (
            "no_duplicate_keys",
            dup_spatial == 0 and dup_baseline == 0,
            f"spatial_dups={dup_spatial} baseline_dups={dup_baseline}",
        )
    )
    checks.append(
        (
            "same_row_count",
            len(spatial) == len(baseline_preds) == 2508,
            f"spatial={len(spatial)} baseline={len(baseline_preds)} expected=2508",
        )
    )
    same_period = (
        spatial["DateTime"].min() == baseline_preds["DateTime"].min()
        and spatial["DateTime"].max() == baseline_preds["DateTime"].max()
    )
    checks.append(
        (
            "same_evaluation_period",
            bool(same_period),
            f"start={spatial['DateTime'].min()} end={spatial['DateTime'].max()}",
        )
    )
    base_pred = baseline_preds["Baseline_GRU_Predicted"].to_numpy(dtype=float)
    spat_pred = spatial["Spatial_GRU_Predicted"].to_numpy(dtype=float)
    checks.append(
        (
            "no_nan_predictions",
            bool(np.isfinite(base_pred).all() and np.isfinite(spat_pred).all())
            and not (np.isnan(base_pred).any() or np.isnan(spat_pred).any()),
            f"baseline_nan={int(np.isnan(base_pred).sum())} spatial_nan={int(np.isnan(spat_pred).sum())}",
        )
    )
    checks.append(
        (
            "no_inf_predictions",
            bool(np.isfinite(base_pred).all() and np.isfinite(spat_pred).all()),
            f"baseline_finite={int(np.isfinite(base_pred).sum())} spatial_finite={int(np.isfinite(spat_pred).sum())}",
        )
    )
    checks.append(("baseline_not_retrained", not used_fit, "fit() was not called"))
    checks.append(
        (
            "spatial_not_retrained",
            not used_fit and not loaded_spatial_model,
            "Spatial GRU weights were not loaded for prediction",
        )
    )
    hashes_ok = hashes_before == hashes_after
    checks.append(
        (
            "protected_hashes_unchanged",
            hashes_ok,
            "baseline/spatial models, scalers, and frozen result files unchanged",
        )
    )
    checks.append(
        (
            "baseline_input_shape",
            baseline_shape == (2508, LOOKBACK, N_FEATURES) and n_features == N_FEATURES,
            f"shape={baseline_shape} n_features={n_features}",
        )
    )
    return checks


def _print_checks(checks: list[tuple[str, bool, str]]) -> bool:
    print("== Same-test-set / safety checks ==")
    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"{status}  {name}: {detail}")
    print("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")
    return all_pass


def main() -> int:
    import keras

    hashes_before = _hash_existing(PROTECTED_MODEL_PATHS + FROZEN_RESULT_PATHS)

    spatial = pd.read_csv(SPATIAL_KEYS_PATH, parse_dates=["DateTime"])
    spatial["Junction"] = spatial["Junction"].astype(int)
    spatial = spatial.sort_values(["Junction", "DateTime"]).reset_index(drop=True)
    if spatial.duplicated(["DateTime", "Junction"]).any():
        raise AssertionError("spatial_gru_comparison_ready.csv has duplicate DateTime/Junction keys")
    if list(spatial.columns[:4]) != ["DateTime", "Junction", "Actual", "Spatial_GRU_Predicted"]:
        raise AssertionError("Unexpected spatial comparison-ready columns")

    test_start = spatial["DateTime"].min()
    test_end = spatial["DateTime"].max()
    n_rows = len(spatial)
    print(
        f"Spatial keys: n={n_rows} period={test_start} -> {test_end} "
        f"junctions={sorted(spatial['Junction'].unique().tolist())}"
    )

    bundle = build_sequences(lookback=LOOKBACK)
    print(
        f"Baseline full sequences: X_test={bundle.X_test.shape} "
        f"n_features={bundle.n_features} lookback={bundle.lookback}"
    )
    if bundle.n_features != N_FEATURES:
        raise AssertionError(f"Baseline n_features must be {N_FEATURES}, got {bundle.n_features}")
    if bundle.X_test.shape != (7122, LOOKBACK, N_FEATURES):
        raise AssertionError(
            f"Expected baseline X_test (7122, {LOOKBACK}, {N_FEATURES}), got {bundle.X_test.shape}"
        )

    meta = bundle.meta_test.copy().reset_index(drop=True)
    meta["DateTime"] = pd.to_datetime(meta["DateTime"])
    meta["Junction"] = meta["Junction"].astype(int)
    meta["row_idx"] = np.arange(len(meta))
    merged = spatial.merge(meta, on=["DateTime", "Junction"], how="left", suffixes=("", "_meta"))
    missing = int(merged["row_idx"].isna().sum())
    if missing:
        raise AssertionError(f"{missing} spatial keys missing from baseline test sequences")
    idx = merged["row_idx"].astype(int).to_numpy()
    X_matched = bundle.X_test[idx]
    y_matched = bundle.y_test[idx]
    print(f"Baseline matched input: X={X_matched.shape}")
    if X_matched.shape != (n_rows, LOOKBACK, N_FEATURES):
        raise AssertionError(f"Expected matched shape {(n_rows, LOOKBACK, N_FEATURES)}, got {X_matched.shape}")

    _, target_scaler = load_scalers()
    model = keras.models.load_model(BASELINE_MODEL_PATH)
    y_pred = predict_vehicles(model, X_matched, target_scaler)
    y_true_baseline = inverse_target(y_matched, target_scaler)
    actual = spatial["Actual"].to_numpy(dtype=np.float64)
    actual_max_diff = float(np.max(np.abs(actual - y_true_baseline)))
    actual_match_ok = bool(np.allclose(actual, y_true_baseline, rtol=0, atol=1e-3))

    baseline_frame = pd.DataFrame(
        {
            "DateTime": spatial["DateTime"],
            "Junction": spatial["Junction"],
            "Actual": actual,
            "Baseline_GRU_Predicted": y_pred,
            "Error": y_pred - actual,
        }
    )
    OUT_BASELINE_PREDS.parent.mkdir(parents=True, exist_ok=True)
    baseline_frame.to_csv(OUT_BASELINE_PREDS, index=False)

    junctions = spatial["Junction"].to_numpy(dtype=int)
    spatial_pred = spatial["Spatial_GRU_Predicted"].to_numpy(dtype=np.float64)
    baseline_metrics = _metrics_by_junction(actual, y_pred, junctions)
    spatial_metrics = _metrics_by_junction(actual, spatial_pred, junctions)

    start_text = str(test_start)
    end_text = str(test_end)
    overall_rows = [
        {
            "Model": "Baseline_GRU",
            **baseline_metrics["Overall"],
            "Test_Start": start_text,
            "Test_End": end_text,
            "Test_Rows": n_rows,
        },
        {
            "Model": "Spatial_GRU",
            **spatial_metrics["Overall"],
            "Test_Start": start_text,
            "Test_End": end_text,
            "Test_Rows": n_rows,
        },
    ]
    _write_overall_csv(overall_rows, OUT_CSV)

    hashes_after = _hash_existing(PROTECTED_MODEL_PATHS + FROZEN_RESULT_PATHS)
    checks = run_checks(
        spatial=spatial,
        baseline_preds=baseline_frame,
        actual_match_ok=actual_match_ok,
        actual_max_diff=actual_max_diff,
        hashes_before=hashes_before,
        hashes_after=hashes_after,
        used_fit=False,
        loaded_spatial_model=False,
        baseline_shape=tuple(X_matched.shape),
        n_features=bundle.n_features,
    )
    checks_ok = _print_checks(checks)

    diffs = {
        key: spatial_metrics["Overall"][key] - baseline_metrics["Overall"][key]
        for key in METRIC_KEYS
    }
    pcts = {
        key: _pct_change(spatial_metrics["Overall"][key], baseline_metrics["Overall"][key])
        for key in METRIC_KEYS
    }
    per_junction = []
    for junction in ("1", "2", "3", "4"):
        per_junction.append(
            {
                "Junction": junction,
                "Baseline_GRU": baseline_metrics[junction],
                "Spatial_GRU": spatial_metrics[junction],
                "diff_Spatial_minus_Baseline": {
                    key: spatial_metrics[junction][key] - baseline_metrics[junction][key]
                    for key in METRIC_KEYS
                },
            }
        )
    payload = {
        "evaluation_period": {"Test_Start": start_text, "Test_End": end_text, "Test_Rows": n_rows},
        "baseline_feature_count": N_FEATURES,
        "spatial_feature_count": SPATIAL_N_FEATURES,
        "lookback": LOOKBACK,
        "baseline_matched_input_shape": list(X_matched.shape),
        "spatial_test_input_shape": [2508, LOOKBACK, SPATIAL_N_FEATURES],
        "overall": overall_rows,
        "per_junction": per_junction,
        "absolute_differences_Spatial_minus_Baseline": diffs,
        "percentage_change_Spatial_vs_Baseline": pcts,
        "no_ranking": True,
        "models_retrained": False,
        "checks": [
            {"name": name, "passed": passed, "detail": detail} for name, passed, detail in checks
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    spatial_shape = (2508, LOOKBACK, SPATIAL_N_FEATURES)
    _write_report(
        OUT_TXT,
        test_start=start_text,
        test_end=end_text,
        n_rows=n_rows,
        baseline_shape=tuple(X_matched.shape),
        spatial_shape=spatial_shape,
        baseline_metrics=baseline_metrics,
        spatial_metrics=spatial_metrics,
        checks=checks,
    )

    print("== Evaluation period ==")
    print(f"A. {start_text} through {end_text}")
    print(f"B. Test rows={n_rows}")
    print(f"C. Baseline matched input shape={tuple(X_matched.shape)} n_features={N_FEATURES} lookback={LOOKBACK}")
    print(f"D. Spatial test input shape={spatial_shape} n_features={SPATIAL_N_FEATURES} lookback={LOOKBACK}")
    print(f"E. Baseline GRU overall: {format_metrics(baseline_metrics['Overall'])}")
    print(f"F. Spatial GRU overall: {format_metrics(spatial_metrics['Overall'])}")
    print("G. Per-junction comparison:")
    for junction in ("1", "2", "3", "4"):
        print(f"   Junction {junction} Baseline: {format_metrics(baseline_metrics[junction])}")
        print(f"   Junction {junction} Spatial:  {format_metrics(spatial_metrics[junction])}")
    print("H. Absolute differences (Spatial - Baseline):")
    for key in METRIC_KEYS:
        print(f"   {key}: {diffs[key]:.6f}")
    print("I. Percentage changes ((Spatial - Baseline) / |Baseline| * 100):")
    for key in METRIC_KEYS:
        print(f"   {key}: {pcts[key]:.4f}%")
    print("J. Same-test-set verification: " + ("PASS" if checks_ok else "FAIL"))
    print(
        "K. Hash/safety checks: "
        + ("PASS" if hashes_before == hashes_after else "FAIL")
    )
    print("L. Files created:")
    print(f"   {OUT_BASELINE_PREDS}")
    print(f"   {OUT_CSV}")
    print(f"   {OUT_JSON}")
    print(f"   {OUT_TXT}")
    print("M. No model was retrained. fit() was not called. Spatial GRU predictions were not regenerated.")
    print("No ranking assigned. results/model_comparison.csv was not updated.")
    return 0 if checks_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
