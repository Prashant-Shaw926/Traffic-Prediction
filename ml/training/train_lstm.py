"""Train the paper LSTM. Default is full training; --smoke is the tiny Junction-1 check."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pandas as pd

from ml.data.load_data import REPO_ROOT
from ml.evaluation.metrics import format_metrics, regression_metrics
from ml.inference.predictor import inverse_target, predict_vehicles
from ml.training.config import (
    BATCH_SIZE,
    DENSE_UNITS,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    HORIZON_STEPS,
    LEARNING_RATE,
    LOOKBACK,
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    MAX_EPOCHS,
    OUTPUT_ACTIVATION,
    SEED,
    SMOKE_EARLY_STOPPING_PATIENCE,
    SMOKE_EPOCHS,
    SMOKE_JUNCTION,
    SMOKE_TRAIN_WINDOWS,
    SMOKE_VAL_WINDOWS,
    TRAIN_SHUFFLE,
)
from ml.training.sequence_generator import (
    ARTIFACTS_DIR,
    SequenceBundle,
    build_sequences,
    load_scalers,
    print_sequence_report,
)

FEATURE_SCALER_PATH = ARTIFACTS_DIR / "feature_scaler.joblib"
TARGET_SCALER_PATH = ARTIFACTS_DIR / "target_scaler.joblib"


def set_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _subset(bundle: SequenceBundle, train_n: int, val_n: int):
    return (
        bundle.X_train[:train_n],
        bundle.y_train[:train_n],
        bundle.meta_train.iloc[:train_n].reset_index(drop=True),
        bundle.X_val[:val_n],
        bundle.y_val[:val_n],
        bundle.meta_val.iloc[:val_n].reset_index(drop=True),
    )


def _save_history_plot(hist: dict[str, list[float]], path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(hist.get("loss", []), label="train")
    plt.plot(hist.get("val_loss", []), label="val")
    plt.xlabel("epoch")
    plt.ylabel("MSE (scaled)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def _metrics_by_junction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = [{"Model": "LSTM", "Junction": "Overall", **regression_metrics(y_true, y_pred)}]
    for junction in sorted(meta["Junction"].unique()):
        mask = meta["Junction"].to_numpy() == junction
        rows.append(
            {
                "Model": "LSTM",
                "Junction": str(int(junction)),
                **regression_metrics(y_true[mask], y_pred[mask]),
            }
        )
    return rows


def _write_metrics(rows: list[dict[str, Any]], json_path: Path, csv_path: Path) -> None:
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["Model", "Junction", "RMSE", "MAE", "MAPE", "R2"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "Model": row["Model"],
                    "Junction": row["Junction"],
                    "RMSE": f"{row['RMSE']:.6f}",
                    "MAE": f"{row['MAE']:.6f}",
                    "MAPE": f"{row['MAPE']:.6f}",
                    "R2": f"{row['R2']:.6f}",
                }
            )


def run_sanity_checks(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_reloaded: np.ndarray,
    meta: pd.DataFrame,
    scaler_hashes_before: dict[str, str],
    scaler_hashes_after: dict[str, str],
    used_test_in_fit: bool,
    used_val_as_train_target: bool,
) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []
    checks.append(
        ("no_nan_predictions", bool(np.isfinite(y_pred).all() and not np.isnan(y_pred).any()),
         f"nan={int(np.isnan(y_pred).sum())} inf={int(np.isinf(y_pred).sum())}")
    )
    checks.append(
        ("no_infinite_predictions", bool(np.isfinite(y_pred).all()),
         f"finite={int(np.isfinite(y_pred).sum())}/{len(y_pred)}")
    )
    checks.append(
        ("equal_lengths", len(y_true) == len(y_pred) == len(meta),
         f"actual={len(y_true)} predicted={len(y_pred)} meta={len(meta)}")
    )
    checks.append(
        ("timestamps_present", "DateTime" in meta.columns and meta["DateTime"].notna().all(),
         "meta DateTime aligned with predictions")
    )
    checks.append(
        ("junction_metadata", "Junction" in meta.columns and meta["Junction"].notna().all(),
         f"junctions={sorted(int(j) for j in meta['Junction'].unique())}")
    )
    checks.append(
        ("test_targets_not_in_training", not used_test_in_fit,
         "fit() received only X_train/y_train and X_val/y_val")
    )
    checks.append(
        ("val_targets_not_in_training_labels", not used_val_as_train_target,
         "y_train does not include validation targets")
    )
    hashes_ok = scaler_hashes_before == scaler_hashes_after
    checks.append(
        ("scalers_not_refit", hashes_ok,
         f"feature={scaler_hashes_after['feature'][:12]} target={scaler_hashes_after['target'][:12]}")
    )
    # Original units: vehicle counts are typically 1-180; scaled would be ~0-1.
    median = float(np.median(y_true))
    checks.append(
        ("predictions_original_units", median > 1.5,
         f"median_actual={median:.2f} (scaled targets would sit near 0-1)")
    )
    reload_ok = np.allclose(y_pred, y_pred_reloaded, atol=1e-5, rtol=1e-5)
    max_diff = float(np.max(np.abs(y_pred - y_pred_reloaded))) if len(y_pred) else 0.0
    checks.append(
        ("reloaded_model_matches", reload_ok, f"max_abs_diff={max_diff:.6g}")
    )
    return checks


def _print_checks(checks: list[tuple[str, bool, str]]) -> bool:
    print("== Sanity checks ==")
    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"{status}  {name}: {detail}")
    print("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")
    return all_pass


def run_smoke() -> int:
    import keras
    from ml.models.lstm import build_lstm

    print(f"KERAS_BACKEND={os.environ.get('KERAS_BACKEND')} keras={keras.__version__}")
    print(
        "Smoke test: Junction 1 only, lookback=168, "
        f"{SMOKE_TRAIN_WINDOWS} train windows, {SMOKE_VAL_WINDOWS} val windows, "
        f"{SMOKE_EPOCHS} epochs. Metrics are NOT a model result."
    )
    bundle = build_sequences(lookback=LOOKBACK, junctions=[SMOKE_JUNCTION])
    print_sequence_report(bundle)
    X_train, y_train, _meta_train, X_val, y_val, meta_val = _subset(
        bundle, SMOKE_TRAIN_WINDOWS, SMOKE_VAL_WINDOWS
    )
    print("== Smoke subset ==")
    print(f"X_train={X_train.shape} y_train={y_train.shape}")
    print(f"X_val={X_val.shape} y_val={y_val.shape}")

    model = build_lstm(lookback=LOOKBACK)
    model.summary()
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=SMOKE_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=False,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=SMOKE_EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    _, target_scaler = load_scalers()
    y_pred = predict_vehicles(model, X_val, target_scaler)
    y_true = inverse_target(y_val, target_scaler)
    metrics = regression_metrics(y_true, y_pred)
    print("== Smoke val metrics (original Vehicles units; not a real result) ==")
    print(format_metrics(metrics))
    print("== Sample predicted vs actual ==")
    for i in range(min(5, len(y_true))):
        row = meta_val.iloc[i]
        print(
            f"{row['DateTime']}  J{int(row['Junction'])}  "
            f"actual={y_true[i]:.2f}  predicted={y_pred[i]:.2f}"
        )
    out_dir = REPO_ROOT / "models" / "trained" / "lstm" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.keras")
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    (out_dir / "config.json").write_text(
        json.dumps({"mode": "smoke", "not_a_real_evaluation": True}, indent=2),
        encoding="utf-8",
    )
    try:
        _save_history_plot(
            hist,
            REPO_ROOT / "results" / "plots" / "lstm_smoke_loss.png",
            "LSTM smoke loss",
        )
    except Exception as exc:
        print(f"Could not save smoke loss plot: {exc}")
    print(f"Saved smoke artifacts under {out_dir}")
    return 0


def run_full() -> int:
    import keras
    from ml.models.lstm import build_lstm

    set_seeds()
    scaler_hashes_before = {
        "feature": _file_sha256(FEATURE_SCALER_PATH),
        "target": _file_sha256(TARGET_SCALER_PATH),
    }

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    print(f"KERAS_BACKEND={os.environ.get('KERAS_BACKEND')} keras={keras.__version__} device={device}")
    print(
        "Full LSTM: all junctions, lookback=168, batch=32, Adam 0.001, "
        f"max_epochs={MAX_EPOCHS}, patience={EARLY_STOPPING_PATIENCE}, "
        f"output={OUTPUT_ACTIVATION}, train_shuffle={TRAIN_SHUFFLE}."
    )
    print(
        "Shuffle applies only to already-built training windows. "
        "Each sample is a self-contained (X ending at T, y=Vehicles(T)) pair, "
        "so batch order does not leak future targets. Test is never passed to fit()."
    )

    bundle = build_sequences(lookback=LOOKBACK)
    print_sequence_report(bundle)
    print(f"X_train={bundle.X_train.shape} X_val={bundle.X_val.shape} X_test={bundle.X_test.shape}")

    model = build_lstm(lookback=LOOKBACK)
    model.summary()

    started = time.perf_counter()
    history = model.fit(
        bundle.X_train,
        bundle.y_train,
        validation_data=(bundle.X_val, bundle.y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=TRAIN_SHUFFLE,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )
    duration_s = time.perf_counter() - started

    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    val_losses = hist.get("val_loss", [])
    train_losses = hist.get("loss", [])
    best_epoch = int(np.argmin(val_losses) + 1) if val_losses else 0
    best_val = val_losses[best_epoch - 1] if val_losses else float("nan")
    epochs_completed = len(train_losses)

    _, target_scaler = load_scalers()
    y_pred = predict_vehicles(model, bundle.X_test, target_scaler)
    y_true = inverse_target(bundle.y_test, target_scaler)

    out_dir = REPO_ROOT / "models" / "trained" / "lstm" / "full"
    pred_dir = REPO_ROOT / "results" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.keras"
    model.save(model_path)

    reloaded = keras.models.load_model(model_path)
    y_pred_reloaded = predict_vehicles(reloaded, bundle.X_test, target_scaler)

    scaler_hashes_after = {
        "feature": _file_sha256(FEATURE_SCALER_PATH),
        "target": _file_sha256(TARGET_SCALER_PATH),
    }

    metric_rows = _metrics_by_junction(y_true, y_pred, bundle.meta_test)
    _write_metrics(
        metric_rows,
        REPO_ROOT / "results" / "lstm_metrics.json",
        REPO_ROOT / "results" / "lstm_metrics.csv",
    )

    pred_frame = pd.DataFrame(
        {
            "DateTime": bundle.meta_test["DateTime"],
            "Junction": bundle.meta_test["Junction"].astype(int),
            "Actual": y_true,
            "Predicted": y_pred,
        }
    ).sort_values(["Junction", "DateTime"])
    pred_path = pred_dir / "lstm_test_predictions.csv"
    pred_frame.to_csv(pred_path, index=False)

    (out_dir / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    config = {
        "mode": "full",
        "lookback": LOOKBACK,
        "horizon_steps": HORIZON_STEPS,
        "n_features": bundle.n_features,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "dropout": DROPOUT,
        "lstm_units": [LSTM_UNITS_1, LSTM_UNITS_2],
        "dense_units": DENSE_UNITS,
        "output_activation": OUTPUT_ACTIVATION,
        "epochs_requested": MAX_EPOCHS,
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "restore_best_weights": True,
        "train_shuffle": TRAIN_SHUFFLE,
        "train_shuffle_rationale": (
            "Independent sequence samples; shuffling batches does not leak future "
            "targets. Validation/test predict without shuffling."
        ),
        "seed": SEED,
        "device": device,
        "keras_backend": os.environ.get("KERAS_BACKEND"),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_duration_seconds": duration_s,
        "scaler_fitted_on": "train (existing artifacts; not refit)",
        "sequence_counts": bundle.counts,
        "test_not_used_in_fit": True,
        "negative_predictions": int((y_pred < 0).sum()),
        "min_predicted": float(np.min(y_pred)),
        "max_predicted": float(np.max(y_pred)),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    plot_path = REPO_ROOT / "results" / "plots" / "lstm_training_history.png"
    _save_history_plot(hist, plot_path, "LSTM training history")

    checks = run_sanity_checks(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_reloaded=y_pred_reloaded,
        meta=bundle.meta_test,
        scaler_hashes_before=scaler_hashes_before,
        scaler_hashes_after=scaler_hashes_after,
        used_test_in_fit=False,
        used_val_as_train_target=False,
    )
    checks_ok = _print_checks(checks)

    overall = next(row for row in metric_rows if row["Junction"] == "Overall")
    print("== Training ==")
    print(f"duration_seconds={duration_s:.1f} ({duration_s / 60:.2f} min)")
    print(f"epochs_completed={epochs_completed} best_epoch={best_epoch}")
    print(f"best_val_loss={best_val:.6f}")
    print(f"final_train_loss={train_losses[-1]:.6f}" if train_losses else "final_train_loss=n/a")
    print(f"final_val_loss={val_losses[-1]:.6f}" if val_losses else "final_val_loss=n/a")
    print("== Overall test (original Vehicles) ==")
    print(format_metrics(overall))
    paper_rmse = overall["RMSE"] < 10
    paper_r2 = overall["R2"] > 0.80
    print(
        f"paper_targets: RMSE<10={'YES' if paper_rmse else 'NO'} "
        f"({overall['RMSE']:.4f}); R2>0.80={'YES' if paper_r2 else 'NO'} ({overall['R2']:.4f})"
    )
    n_neg = int((y_pred < 0).sum())
    print(f"negative_predictions={n_neg} min_predicted={float(np.min(y_pred)):.4f}")
    print("== Per-junction test ==")
    for row in metric_rows:
        if row["Junction"] == "Overall":
            continue
        print(f"Junction {row['Junction']}: {format_metrics(row)}")
    print("== Sample actual vs predicted ==")
    sample = pred_frame.head(8)
    for rec in sample.itertuples(index=False):
        print(
            f"{rec.DateTime} | Junction {int(rec.Junction)} | "
            f"Actual {rec.Actual:.2f} | Predicted {rec.Predicted:.2f}"
        )
    print("== Artifacts ==")
    print(model_path)
    print(out_dir / "history.json")
    print(out_dir / "config.json")
    print(REPO_ROOT / "results" / "lstm_metrics.csv")
    print(pred_path)
    print(plot_path)
    return 0 if checks_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Junction-1, 256/64 windows, 2 epochs. Not a real evaluation.",
    )
    args = parser.parse_args()
    set_seeds()
    if args.smoke:
        return run_smoke()
    return run_full()


if __name__ == "__main__":
    raise SystemExit(main())
