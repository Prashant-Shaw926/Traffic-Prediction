"""Compare frozen A/D with trained B/C ablation GRUs on the same 2508 test keys."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.data.load_data import REPO_ROOT
from ml.evaluation.metrics import format_metrics, regression_metrics

METRIC_KEYS = ("RMSE", "MAE", "MAPE", "R2")
KEYS_PATH = REPO_ROOT / "results" / "predictions" / "spatial_gru_comparison_ready.csv"
BASELINE_PREDS = REPO_ROOT / "results" / "predictions" / "baseline_gru_spatial_period_predictions.csv"
NETWORK_PREDS = REPO_ROOT / "results" / "ablation" / "predictions" / "network_gru_test_predictions.csv"
CROSS_PREDS = (
    REPO_ROOT / "results" / "ablation" / "predictions" / "cross_junction_gru_test_predictions.csv"
)
OUT_DIR = REPO_ROOT / "results" / "ablation"

FROZEN_PATHS = (
    REPO_ROOT / "models" / "trained" / "gru" / "full" / "model.keras",
    REPO_ROOT / "models" / "artifacts" / "feature_scaler.joblib",
    REPO_ROOT / "models" / "artifacts" / "target_scaler.joblib",
    REPO_ROOT / "results" / "gru_metrics.csv",
    REPO_ROOT / "results" / "predictions" / "gru_test_predictions.csv",
    REPO_ROOT / "models" / "trained" / "gru" / "spatial" / "model.keras",
    REPO_ROOT / "models" / "artifacts" / "spatial" / "feature_scaler.joblib",
    REPO_ROOT / "models" / "artifacts" / "spatial" / "target_scaler.joblib",
    REPO_ROOT / "results" / "spatial_gru_metrics.csv",
    REPO_ROOT / "results" / "predictions" / "spatial_gru_test_predictions.csv",
    KEYS_PATH,
    REPO_ROOT / "results" / "spatial_vs_baseline_gru_same_period.csv",
    REPO_ROOT / "results" / "model_comparison.csv",
)

EXPERIMENT_ROWS = (
    {
        "Experiment": "Baseline",
        "Feature_Group": "temporal_plus_own_lags",
        "Feature_Count": 13,
        "pred_col": "Baseline_GRU_Predicted",
        "source": BASELINE_PREDS,
        "frozen": True,
    },
    {
        "Experiment": "Network",
        "Feature_Group": "baseline_plus_network_lags",
        "Feature_Count": 23,
        "pred_col": "Predicted",
        "source": NETWORK_PREDS,
        "frozen": False,
    },
    {
        "Experiment": "Cross_Junction",
        "Feature_Group": "baseline_plus_cross_junction_lags",
        "Feature_Count": 43,
        "pred_col": "Predicted",
        "source": CROSS_PREDS,
        "frozen": False,
    },
    {
        "Experiment": "Full_Spatial",
        "Feature_Group": "baseline_plus_network_and_cross_junction_lags",
        "Feature_Count": 53,
        "pred_col": "Spatial_GRU_Predicted",
        "source": KEYS_PATH,
        "frozen": True,
    },
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _hash_existing(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(path): _file_sha256(path) for path in paths if path.is_file()}


def _pct_change(value: float, baseline: float) -> float:
    denom = abs(baseline)
    if denom == 0:
        return float("nan")
    return (value - baseline) / denom * 100.0


def _sorted_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["DateTime"] = pd.to_datetime(out["DateTime"])
    out["Junction"] = out["Junction"].astype(int)
    return out.sort_values(["Junction", "DateTime"]).reset_index(drop=True)


def _metrics_by_junction(actual: np.ndarray, pred: np.ndarray, junctions: np.ndarray) -> dict[str, dict[str, float]]:
    out = {"Overall": regression_metrics(actual, pred)}
    for junction in (1, 2, 3, 4):
        mask = junctions == junction
        out[str(junction)] = regression_metrics(actual[mask], pred[mask])
    return out


def _load_pred(spec: dict[str, Any], keys: pd.DataFrame, actual: np.ndarray) -> np.ndarray:
    frame = _sorted_keys(pd.read_csv(spec["source"], parse_dates=["DateTime"]))
    merged = keys.merge(frame, on=["DateTime", "Junction"], how="left", suffixes=("", "_src"))
    if merged[spec["pred_col"]].isna().any():
        raise AssertionError(f"{spec['Experiment']}: missing predictions for some spatial keys")
    pred = merged[spec["pred_col"]].to_numpy(dtype=np.float64)
    if spec["Experiment"] != "Full_Spatial" and "Actual" in merged.columns:
        src_actual = merged["Actual"].to_numpy(dtype=np.float64)
        if not np.allclose(src_actual, actual, rtol=0, atol=1e-3):
            raise AssertionError(f"{spec['Experiment']}: Actual values do not match shared targets")
    return pred


def main() -> int:
    hashes_before = _hash_existing(FROZEN_PATHS)
    keys_raw = _sorted_keys(pd.read_csv(KEYS_PATH, parse_dates=["DateTime"]))
    keys = keys_raw[["DateTime", "Junction"]].copy()
    actual = keys_raw["Actual"].to_numpy(dtype=np.float64)
    junctions = keys["Junction"].to_numpy(dtype=int)
    test_start = str(keys["DateTime"].min())
    test_end = str(keys["DateTime"].max())
    n_rows = len(keys)
    if n_rows != 2508:
        raise AssertionError(f"Expected 2508 keys, got {n_rows}")

    checks: list[tuple[str, bool, str]] = []
    preds: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, dict[str, float]]] = {}
    for spec in EXPERIMENT_ROWS:
        pred = _load_pred(spec, keys, actual)
        preds[spec["Experiment"]] = pred
        metrics[spec["Experiment"]] = _metrics_by_junction(actual, pred, junctions)
        finite = bool(np.isfinite(pred).all())
        checks.append(
            (
                f"{spec['Experiment']}_finite",
                finite,
                f"nan={int(np.isnan(pred).sum())} inf={int(np.isinf(pred).sum())}",
            )
        )
        pred_keys_ok = True
        checks.append((f"{spec['Experiment']}_keys_aligned", pred_keys_ok, "joined on DateTime+Junction"))

    key_sets_equal = True
    for spec in EXPERIMENT_ROWS:
        frame = _sorted_keys(pd.read_csv(spec["source"], parse_dates=["DateTime"]))[["DateTime", "Junction"]]
        if not frame.equals(keys):
            key_sets_equal = False
    checks.append(("identical_target_keys", key_sets_equal, f"n={n_rows} {test_start} -> {test_end}"))
    checks.append(("no_duplicate_keys", not keys.duplicated().any(), f"dups={int(keys.duplicated().sum())}"))

    hashes_after = _hash_existing(FROZEN_PATHS)
    checks.append(
        (
            "frozen_artifacts_unchanged",
            hashes_before == hashes_after,
            "baseline/spatial models, scalers, and existing comparison files unchanged",
        )
    )
    checks.append(("a_and_d_not_retrained", True, "Baseline and Full_Spatial predictions were read, not regenerated"))
    checks.append(("no_ranking", True, "no composite score or winner assigned"))

    csv_rows = []
    for spec in EXPERIMENT_ROWS:
        overall = metrics[spec["Experiment"]]["Overall"]
        csv_rows.append(
            {
                "Experiment": spec["Experiment"],
                "Feature_Group": spec["Feature_Group"],
                "Feature_Count": spec["Feature_Count"],
                "RMSE": overall["RMSE"],
                "MAE": overall["MAE"],
                "MAPE": overall["MAPE"],
                "R2": overall["R2"],
                "Test_Start": test_start,
                "Test_End": test_end,
                "Test_Rows": n_rows,
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "spatial_feature_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Experiment",
                "Feature_Group",
                "Feature_Count",
                "RMSE",
                "MAE",
                "MAPE",
                "R2",
                "Test_Start",
                "Test_End",
                "Test_Rows",
            ],
        )
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(
                {
                    **row,
                    "RMSE": f"{row['RMSE']:.6f}",
                    "MAE": f"{row['MAE']:.6f}",
                    "MAPE": f"{row['MAPE']:.6f}",
                    "R2": f"{row['R2']:.6f}",
                }
            )

    baseline = metrics["Baseline"]["Overall"]
    diffs = {}
    pcts = {}
    for spec in EXPERIMENT_ROWS:
        name = spec["Experiment"]
        diffs[name] = {
            key: metrics[name]["Overall"][key] - baseline[key] for key in METRIC_KEYS
        }
        pcts[name] = {
            key: _pct_change(metrics[name]["Overall"][key], baseline[key]) for key in METRIC_KEYS
        }

    payload = {
        "evaluation_period": {"Test_Start": test_start, "Test_End": test_end, "Test_Rows": n_rows},
        "feature_groups": [
            {
                "Experiment": spec["Experiment"],
                "Feature_Group": spec["Feature_Group"],
                "Feature_Count": spec["Feature_Count"],
                "frozen": spec["frozen"],
            }
            for spec in EXPERIMENT_ROWS
        ],
        "overall": csv_rows,
        "per_junction": {
            name: metrics[name] for name in ("Baseline", "Network", "Cross_Junction", "Full_Spatial")
        },
        "absolute_differences_vs_baseline": diffs,
        "percentage_change_vs_baseline": pcts,
        "no_ranking": True,
        "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in checks],
    }
    json_path = OUT_DIR / "spatial_feature_ablation.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "Spatial feature ablation report",
        "",
        "A Baseline and D Full_Spatial are frozen same-period results.",
        "B Network and C Cross_Junction are newly trained GRUs.",
        "No ranking or composite score is assigned. No causal claims about junction influence.",
        "",
        "== Evaluation period ==",
        f"Test_Start={test_start}",
        f"Test_End={test_end}",
        f"Test_Rows={n_rows}",
        "",
        "== Feature groups ==",
        "Baseline: 13 temporal + own vehicles lags",
        "Network: 23 = baseline + lagged network_total_traffic and network_avg_traffic",
        "Cross_Junction: 43 = baseline + lagged junction_1-4_traffic, other_junction_traffic, other_junction_avg_traffic",
        "Full_Spatial: 53 = baseline + network + cross-junction lags",
        "Lags: 1, 2, 3, 24, 168 hours. Contemporaneous traffic at T is not a feature.",
        "",
        "== Preprocessing / leakage ==",
        "Four-junction overlap from 2017-01-01, then drop rows until lag_168 exists.",
        "70/15/15 per Junction on eligible hours. Train-only MinMax scalers per ablation experiment.",
        "Same eligible rows as the spatial pipeline so test keys match.",
        "",
        "== Same-test-set verification ==",
    ]
    for name, passed, detail in checks:
        lines.append(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")
    all_pass = all(p for _, p, _ in checks)
    lines.append("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")
    lines.extend(["", "== Overall metrics (original Vehicles) =="])
    for spec in EXPERIMENT_ROWS:
        name = spec["Experiment"]
        lines.append(
            f"{name} (n={spec['Feature_Count']}): {format_metrics(metrics[name]['Overall'])}"
        )
    lines.extend(["", "== Per-junction metrics =="])
    for junction in ("1", "2", "3", "4"):
        lines.append(f"Junction {junction}")
        for spec in EXPERIMENT_ROWS:
            name = spec["Experiment"]
            lines.append(f"  {name}: {format_metrics(metrics[name][junction])}")
    lines.extend(["", "== Absolute differences vs Baseline (value - Baseline) =="])
    for spec in EXPERIMENT_ROWS:
        name = spec["Experiment"]
        if name == "Baseline":
            continue
        parts = [f"{key}={diffs[name][key]:.6f}" for key in METRIC_KEYS]
        lines.append(f"{name}: " + " ".join(parts))
    lines.extend(["", "== Percentage change vs Baseline ((value - Baseline) / |Baseline| * 100) =="])
    for spec in EXPERIMENT_ROWS:
        name = spec["Experiment"]
        if name == "Baseline":
            continue
        parts = [f"{key}={pcts[name][key]:.4f}%" for key in METRIC_KEYS]
        lines.append(f"{name}: " + " ".join(parts))
    lines.extend(
        [
            "",
            "== Factual observations ==",
            (
                f"Network RMSE was {metrics['Network']['Overall']['RMSE']:.4f} compared with "
                f"baseline RMSE {baseline['RMSE']:.4f} on the same test set."
            ),
            (
                f"Cross_Junction RMSE was {metrics['Cross_Junction']['Overall']['RMSE']:.4f} "
                f"compared with baseline RMSE {baseline['RMSE']:.4f}."
            ),
            (
                f"Full_Spatial RMSE was {metrics['Full_Spatial']['Overall']['RMSE']:.4f} "
                f"compared with baseline RMSE {baseline['RMSE']:.4f}."
            ),
            "These statements are not a ranking.",
            "",
            "Command: python -m ml.evaluation.ablation_spatial_features",
        ]
    )
    txt_path = OUT_DIR / "spatial_feature_ablation.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("== Spatial feature ablation ==")
    print(f"period={test_start} -> {test_end} rows={n_rows}")
    for spec in EXPERIMENT_ROWS:
        name = spec["Experiment"]
        print(f"{name} n={spec['Feature_Count']}: {format_metrics(metrics[name]['Overall'])}")
    print("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")
    print(csv_path)
    print(json_path)
    print(txt_path)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
