"""Train the paper CNN-LSTM on the same sequences and scalers as LSTM/GRU.

Does not touch LSTM or GRU artifacts. Does not write model_comparison.csv.
"""

from __future__ import annotations

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
    CNN_LSTM_UNITS,
    CONV_FILTERS,
    CONV_KERNEL_SIZE,
    DENSE_UNITS,
    EARLY_STOPPING_PATIENCE,
    HORIZON_STEPS,
    LEARNING_RATE,
    LOOKBACK,
    MAX_EPOCHS,
    OUTPUT_ACTIVATION,
    SEED,
    TRAIN_SHUFFLE,
)
from ml.training.sequence_generator import (
    ARTIFACTS_DIR,
    build_sequences,
    load_scalers,
    print_sequence_report,
)

FEATURE_SCALER_PATH = ARTIFACTS_DIR / "feature_scaler.joblib"
TARGET_SCALER_PATH = ARTIFACTS_DIR / "target_scaler.joblib"
MODEL_NAME = "CNN-LSTM"


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
    rows = [{"Model": MODEL_NAME, "Junction": "Overall", **regression_metrics(y_true, y_pred)}]
    for junction in sorted(meta["Junction"].unique()):
        mask = meta["Junction"].to_numpy() == junction
        rows.append(
            {
                "Model": MODEL_NAME,
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
        (
            "no_nan_predictions",
            bool(np.isfinite(y_pred).all() and not np.isnan(y_pred).any()),
            f"nan={int(np.isnan(y_pred).sum())} inf={int(np.isinf(y_pred).sum())}",
        )
    )
    checks.append(
        (
            "no_infinite_predictions",
            bool(np.isfinite(y_pred).all()),
            f"finite={int(np.isfinite(y_pred).sum())}/{len(y_pred)}",
        )
    )
    checks.append(
        (
            "equal_lengths",
            len(y_true) == len(y_pred) == len(meta),
            f"actual={len(y_true)} predicted={len(y_pred)} meta={len(meta)}",
        )
    )
    checks.append(
        (
            "timestamps_present",
            "DateTime" in meta.columns and meta["DateTime"].notna().all(),
            "meta DateTime aligned with predictions",
        )
    )
    checks.append(
        (
            "junction_metadata",
            "Junction" in meta.columns and meta["Junction"].notna().all(),
            f"junctions={sorted(int(j) for j in meta['Junction'].unique())}",
        )
    )
    checks.append(
        (
            "test_targets_not_in_training",
            not used_test_in_fit,
            "fit() received only X_train/y_train and X_val/y_val",
        )
    )
    checks.append(
        (
            "val_targets_not_in_training_labels",
            not used_val_as_train_target,
            "y_train does not include validation targets",
        )
    )
    hashes_ok = scaler_hashes_before == scaler_hashes_after
    checks.append(
        (
            "scalers_not_refit",
            hashes_ok,
            f"feature={scaler_hashes_after['feature'][:12]} target={scaler_hashes_after['target'][:12]}",
        )
    )
    median = float(np.median(y_true))
    checks.append(
        (
            "predictions_original_units",
            median > 1.5,
            f"median_actual={median:.2f} (scaled targets would sit near 0-1)",
        )
    )
    reload_ok = np.allclose(y_pred, y_pred_reloaded, atol=1e-5, rtol=1e-5)
    max_diff = float(np.max(np.abs(y_pred - y_pred_reloaded))) if len(y_pred) else 0.0
    checks.append(("reloaded_model_matches", reload_ok, f"max_abs_diff={max_diff:.6g}"))
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


def main() -> int:
    import keras
    from ml.models.cnn_lstm import build_cnn_lstm

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
        "Full CNN-LSTM: all junctions, lookback=168, batch=32, Adam 0.001, "
        f"Conv1D filters={CONV_FILTERS} kernel={CONV_KERNEL_SIZE}, LSTM={CNN_LSTM_UNITS}, "
        f"max_epochs={MAX_EPOCHS}, patience={EARLY_STOPPING_PATIENCE}, "
        f"output={OUTPUT_ACTIVATION}, train_shuffle={TRAIN_SHUFFLE}."
    )
    print(
        "Shuffle applies only to already-built training windows. "
        "Test is never passed to fit(). LSTM and GRU artifacts are not modified."
    )

    bundle = build_sequences(lookback=LOOKBACK)
    print_sequence_report(bundle)
    print(f"X_train={bundle.X_train.shape} X_val={bundle.X_val.shape} X_test={bundle.X_test.shape}")

    model = build_cnn_lstm(lookback=LOOKBACK)
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

    out_dir = REPO_ROOT / "models" / "trained" / "cnn_lstm" / "full"
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
        REPO_ROOT / "results" / "cnn_lstm_metrics.json",
        REPO_ROOT / "results" / "cnn_lstm_metrics.csv",
    )

    pred_frame = pd.DataFrame(
        {
            "DateTime": bundle.meta_test["DateTime"],
            "Junction": bundle.meta_test["Junction"].astype(int),
            "Actual": y_true,
            "Predicted": y_pred,
        }
    ).sort_values(["Junction", "DateTime"])
    pred_path = pred_dir / "cnn_lstm_test_predictions.csv"
    pred_frame.to_csv(pred_path, index=False)

    (out_dir / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    config = {
        "mode": "full",
        "model": MODEL_NAME,
        "lookback": LOOKBACK,
        "horizon_steps": HORIZON_STEPS,
        "n_features": bundle.n_features,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "conv_filters": CONV_FILTERS,
        "conv_kernel_size": CONV_KERNEL_SIZE,
        "lstm_units": CNN_LSTM_UNITS,
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

    plot_path = REPO_ROOT / "results" / "plots" / "cnn_lstm_training_history.png"
    _save_history_plot(hist, plot_path, "CNN-LSTM training history")

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
    print(REPO_ROOT / "results" / "cnn_lstm_metrics.csv")
    print(pred_path)
    print(plot_path)
    return 0 if checks_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
