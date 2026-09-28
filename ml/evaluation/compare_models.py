"""Compare existing LSTM, GRU, and CNN-LSTM test metrics. Does not retrain."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from ml.data.load_data import REPO_ROOT

METRIC_KEYS = ("RMSE", "MAE", "MAPE", "R2")
MODEL_ORDER = ("LSTM", "GRU", "CNN-LSTM")
CSV_DECIMALS = 6

SOURCES = {
    "LSTM": {
        "metrics_json": REPO_ROOT / "results" / "lstm_metrics.json",
        "metrics_csv": REPO_ROOT / "results" / "lstm_metrics.csv",
        "config": REPO_ROOT / "models" / "trained" / "lstm" / "full" / "config.json",
        "history": REPO_ROOT / "models" / "trained" / "lstm" / "full" / "history.json",
    },
    "GRU": {
        "metrics_json": REPO_ROOT / "results" / "gru_metrics.json",
        "metrics_csv": REPO_ROOT / "results" / "gru_metrics.csv",
        "config": REPO_ROOT / "models" / "trained" / "gru" / "full" / "config.json",
        "history": REPO_ROOT / "models" / "trained" / "gru" / "full" / "history.json",
    },
    "CNN-LSTM": {
        "metrics_json": REPO_ROOT / "results" / "cnn_lstm_metrics.json",
        "metrics_csv": REPO_ROOT / "results" / "cnn_lstm_metrics.csv",
        "config": REPO_ROOT / "models" / "trained" / "cnn_lstm" / "full" / "config.json",
        "history": REPO_ROOT / "models" / "trained" / "cnn_lstm" / "full" / "history.json",
    },
}

OUT_CSV = REPO_ROOT / "results" / "model_comparison.csv"
OUT_JSON = REPO_ROOT / "results" / "model_comparison.json"
OUT_REPORT = REPO_ROOT / "results" / "model_comparison_report.txt"
OUT_PLOT = REPO_ROOT / "results" / "plots" / "model_comparison.png"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_by_junction(rows: list[dict[str, Any]], junction: str) -> dict[str, Any]:
    for row in rows:
        if str(row["Junction"]) == junction:
            return row
    raise KeyError(f"Junction {junction!r} not found")


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _fmt6(value: float) -> str:
    return f"{value:.{CSV_DECIMALS}f}"


def _best_val_loss(history: dict[str, list[float]]) -> float | None:
    val_losses = history.get("val_loss") or []
    if not val_losses:
        return None
    return float(min(val_losses))


def _load_model(name: str) -> dict[str, Any]:
    paths = SOURCES[name]
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source artifacts:\n  " + "\n  ".join(missing))

    metric_rows = _load_json(paths["metrics_json"])
    csv_rows = _load_csv_rows(paths["metrics_csv"])
    config = _load_json(paths["config"])
    history = _load_json(paths["history"])
    overall = _row_by_junction(metric_rows, "Overall")
    overall_csv = _row_by_junction(csv_rows, "Overall")

    record = {
        "Model": name,
        "RMSE": float(overall["RMSE"]),
        "MAE": float(overall["MAE"]),
        "MAPE": float(overall["MAPE"]),
        "R2": float(overall["R2"]),
        "training_duration_seconds": config.get("training_duration_seconds"),
        "best_epoch": config.get("best_epoch"),
        "epochs_completed": config.get("epochs_completed"),
        "best_val_loss": _best_val_loss(history),
        "per_junction": [
            {
                "Junction": str(row["Junction"]),
                "RMSE": float(row["RMSE"]),
                "MAE": float(row["MAE"]),
                "MAPE": float(row["MAPE"]),
                "R2": float(row["R2"]),
            }
            for row in metric_rows
            if str(row["Junction"]) != "Overall"
        ],
        "source_csv_overall": {
            key: overall_csv[key] for key in ("Model", "RMSE", "MAE", "MAPE", "R2")
        },
        "source_json_path": str(paths["metrics_json"]),
        "source_csv_path": str(paths["metrics_csv"]),
    }
    record["per_junction"].sort(key=lambda row: int(row["Junction"]))
    return record


def _arg_extreme(records: list[dict[str, Any]], key: str, *, lowest: bool) -> str:
    ranked = sorted(records, key=lambda row: row[key], reverse=not lowest)
    winner = ranked[0][key]
    tied = [row["Model"] for row in records if row[key] == winner]
    return ", ".join(tied)


def _duration_minutes(seconds: Any) -> str:
    if seconds is None:
        return "n/a"
    return f"{float(seconds) / 60.0:.2f}"


def _write_comparison_csv(records: list[dict[str, Any]]) -> None:
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Model", "RMSE", "MAE", "MAPE", "R2"])
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    "Model": row["Model"],
                    "RMSE": _fmt6(row["RMSE"]),
                    "MAE": _fmt6(row["MAE"]),
                    "MAPE": _fmt6(row["MAPE"]),
                    "R2": _fmt6(row["R2"]),
                }
            )


def _write_comparison_json(records: list[dict[str, Any]]) -> None:
    payload = [
        {
            "Model": row["Model"],
            "RMSE": row["RMSE"],
            "MAE": row["MAE"],
            "MAPE": row["MAPE"],
            "R2": row["R2"],
            "training_duration_seconds": row["training_duration_seconds"],
            "best_epoch": row["best_epoch"],
            "best_val_loss": row["best_val_loss"],
            "epochs_completed": row["epochs_completed"],
        }
        for row in records
    ]
    OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _metric_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    def fmt(row: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
    lines = [fmt(headers), fmt(["-" * w for w in widths])]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def _write_report(records: list[dict[str, Any]]) -> None:
    overall_rows = [
        [
            row["Model"],
            f"{row['RMSE']:.6f}",
            f"{row['MAE']:.6f}",
            f"{row['MAPE']:.6f}",
            f"{row['R2']:.6f}",
        ]
        for row in records
    ]
    junction_ids = sorted(
        {int(item["Junction"]) for row in records for item in row["per_junction"]}
    )
    junction_sections: list[str] = []
    for junction in junction_ids:
        j_rows = []
        for row in records:
            match = next(
                item for item in row["per_junction"] if int(item["Junction"]) == junction
            )
            j_rows.append(
                [
                    row["Model"],
                    f"{match['RMSE']:.6f}",
                    f"{match['MAE']:.6f}",
                    f"{match['MAPE']:.6f}",
                    f"{match['R2']:.6f}",
                ]
            )
        junction_sections.append(
            f"Junction {junction}\n"
            + _metric_table(["Model", "RMSE", "MAE", "MAPE", "R2"], j_rows)
        )

    duration_rows = [
        [
            row["Model"],
            "n/a" if row["training_duration_seconds"] is None else f"{row['training_duration_seconds']:.1f}",
            _duration_minutes(row["training_duration_seconds"]),
            "n/a" if row["epochs_completed"] is None else str(row["epochs_completed"]),
            "n/a" if row["best_epoch"] is None else str(row["best_epoch"]),
            "n/a" if row["best_val_loss"] is None else f"{row['best_val_loss']:.6f}",
        ]
        for row in records
    ]

    lowest_rmse = _arg_extreme(records, "RMSE", lowest=True)
    lowest_mae = _arg_extreme(records, "MAE", lowest=True)
    lowest_mape = _arg_extreme(records, "MAPE", lowest=True)
    highest_r2 = _arg_extreme(records, "R2", lowest=False)

    text = f"""Deep-learning model comparison
==============================

This report reads existing evaluation artifacts. No models were retrained.
ARIMA is not included. No composite score or overall ranking is assigned.

1. Dataset / evaluation consistency
-----------------------------------
All three models used the same verified pipeline:

- Test set: 7122 one-step-ahead windows (same timestamps and junctions)
- Lookback: 168 hours
- Features: 13 (hour, day_of_week, day, month, is_weekend, hour_sin,
  hour_cos, Junction, vehicles_lag_1/2/3/24/168)
- Scalers: existing train-only MinMax feature_scaler.joblib and
  target_scaler.joblib (not refit for any of these experiments)
- Target: Vehicles at timestamp T, original units after inverse scaling
- Metrics: RMSE, MAE, MAPE (%), R2 via ml/evaluation/metrics.py
- Train / validation / test sequence counts: 32543 / 7115 / 7122

2. Overall results (test set, original Vehicles)
------------------------------------------------
{_metric_table(["Model", "RMSE", "MAE", "MAPE", "R2"], overall_rows)}

RMSE, MAE, and MAPE are lower-is-better. R2 is higher-is-better.

3. Per-junction comparison
--------------------------
{"\n\n".join(junction_sections)}

4. Factual metric observations (overall test set only)
------------------------------------------------------
- Lowest RMSE: {lowest_rmse}
- Lowest MAE: {lowest_mae}
- Lowest MAPE: {lowest_mape}
- Highest R2: {highest_r2}

These are separate observations per metric. This report does not assign
an overall ranking or a combined score.

5. Scope
--------
These observations apply only to this experiment and this held-out test
set. They should not be generalized to other datasets, cities, horizons,
or future time periods.

6. Training duration (from saved configs / histories)
-----------------------------------------------------
{_metric_table(["Model", "Duration_s", "Duration_min", "Epochs", "Best_epoch", "Best_val_loss"], duration_rows)}

7. Retraining
-------------
No LSTM, GRU, or CNN-LSTM weights were loaded for training or updated.
This script only read existing metric, config, and history files and
wrote comparison outputs under results/.
"""
    OUT_REPORT.write_text(text, encoding="utf-8")


def _save_plot(records: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    names = [row["Model"] for row in records]
    panels = [
        ("RMSE", "RMSE (lower is better)", True),
        ("MAE", "MAE (lower is better)", True),
        ("MAPE", "MAPE % (lower is better)", True),
        ("R2", "R2 (higher is better)", False),
    ]
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    x = np.arange(len(names))
    for ax, (key, title, _lower) in zip(axes.ravel(), panels):
        values = [row[key] for row in records]
        bars = ax.bar(x, values, color=colors, width=0.65)
        ax.set_xticks(x, names)
        ax.set_title(title)
        ax.set_ylabel(key)
        ymax = max(values)
        ymin = min(values)
        pad = (ymax - ymin) * 0.12 if ymax != ymin else abs(ymax) * 0.05 or 0.05
        ax.set_ylim(0 if key != "R2" else max(0.0, ymin - pad), ymax + pad)
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.3f}" if key != "R2" else f"{value:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.suptitle("LSTM vs GRU vs CNN-LSTM test metrics (original Vehicles)")
    fig.tight_layout()
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOT, dpi=120)
    plt.close(fig)


def _validate(records: list[dict[str, Any]]) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    source_ok = all(
        SOURCES[name]["metrics_json"].is_file()
        and SOURCES[name]["metrics_csv"].is_file()
        for name in MODEL_ORDER
    )
    checks.append(("source_metric_files_exist", source_ok, "lstm/gru/cnn_lstm metrics json+csv"))

    written_csv = _load_csv_rows(OUT_CSV)
    models = [row["Model"] for row in written_csv]
    checks.append(
        (
            "comparison_csv_three_models",
            models == list(MODEL_ORDER),
            f"models={models}",
        )
    )

    written_json = _load_json(OUT_JSON)
    all_finite = True
    for row in written_json:
        for key in METRIC_KEYS:
            if not _finite(float(row[key])):
                all_finite = False
    checks.append(("no_nan_inf", all_finite, "overall RMSE/MAE/MAPE/R2 finite"))

    json_match = True
    csv_match = True
    details: list[str] = []
    for record, json_row, csv_row in zip(records, written_json, written_csv):
        for key in METRIC_KEYS:
            if float(json_row[key]) != record[key]:
                json_match = False
                details.append(f"{record['Model']} {key} json mismatch")
            if csv_row[key] != _fmt6(record[key]):
                csv_match = False
                details.append(f"{record['Model']} {key} csv mismatch")
            if record["source_csv_overall"][key] != _fmt6(record[key]):
                csv_match = False
                details.append(f"{record['Model']} {key} source csv mismatch")
    checks.append(("json_matches_source_metrics", json_match, "overall floats equal source json"))
    checks.append(
        (
            "csv_matches_source_rounding",
            csv_match,
            "six-decimal overall values match source csv",
        )
    )
    if details:
        checks.append(("mismatch_detail", False, "; ".join(details[:8])))
    return checks


def main() -> int:
    records = [_load_model(name) for name in MODEL_ORDER]
    _write_comparison_csv(records)
    _write_comparison_json(records)
    _write_report(records)
    _save_plot(records)
    checks = _validate(records)

    print("== Validation ==")
    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"{status}  {name}: {detail}")
    print("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")

    print("== Overall comparison (original Vehicles) ==")
    print("Model      RMSE      MAE       MAPE      R2")
    for row in records:
        print(
            f"{row['Model']:<10} {_fmt6(row['RMSE'])}  {_fmt6(row['MAE'])}  "
            f"{_fmt6(row['MAPE'])}  {_fmt6(row['R2'])}"
        )
    print("== Observations ==")
    print(f"lowest RMSE: {_arg_extreme(records, 'RMSE', lowest=True)}")
    print(f"lowest MAE: {_arg_extreme(records, 'MAE', lowest=True)}")
    print(f"lowest MAPE: {_arg_extreme(records, 'MAPE', lowest=True)}")
    print(f"highest R2: {_arg_extreme(records, 'R2', lowest=False)}")
    print("No overall ranking assigned. No models retrained.")
    print("== Artifacts ==")
    print(OUT_CSV)
    print(OUT_JSON)
    print(OUT_REPORT)
    print(OUT_PLOT)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
