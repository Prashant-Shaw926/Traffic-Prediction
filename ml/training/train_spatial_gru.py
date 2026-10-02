"""Train spatial/network GRU. Does not touch baseline GRU artifacts or results."""

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
from ml.features.spatial_features import SPATIAL_FEATURE_COLUMNS
from ml.inference.predictor import inverse_target, predict_vehicles
from ml.training.config import (
    BATCH_SIZE,
    DENSE_UNITS,
    DROPOUT,
    EARLY_STOPPING_PATIENCE,
    GRU_UNITS_1,
    GRU_UNITS_2,
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

SPATIAL_N_FEATURES = 53
SPATIAL_PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "spatial"
SPATIAL_ARTIFACTS_DIR = REPO_ROOT / "models" / "artifacts" / "spatial"
SPATIAL_FEATURE_CONFIG = SPATIAL_ARTIFACTS_DIR / "feature_config.json"
SPATIAL_FEATURE_SCALER = SPATIAL_ARTIFACTS_DIR / "feature_scaler.joblib"
SPATIAL_TARGET_SCALER = SPATIAL_ARTIFACTS_DIR / "target_scaler.joblib"
SPATIAL_MODEL_DIR = REPO_ROOT / "models" / "trained" / "gru" / "spatial"
BASELINE_FEATURE_SCALER = ARTIFACTS_DIR / "feature_scaler.joblib"
BASELINE_TARGET_SCALER = ARTIFACTS_DIR / "target_scaler.joblib"
BASELINE_GRU_MODEL = REPO_ROOT / "models" / "trained" / "gru" / "full" / "model.keras"
BASELINE_GRU_METRICS = REPO_ROOT / "results" / "gru_metrics.csv"
BASELINE_GRU_PREDICTIONS = REPO_ROOT / "results" / "predictions" / "gru_test_predictions.csv"

PROTECTED_BASELINE_PATHS = (
    BASELINE_FEATURE_SCALER,
    BASELINE_TARGET_SCALER,
    BASELINE_GRU_MODEL,
    BASELINE_GRU_METRICS,
    BASELINE_GRU_PREDICTIONS,
)


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


def _hash_paths(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(path): _file_sha256(path) for path in paths}


def load_spatial_feature_columns() -> list[str]:
    config = json.loads(SPATIAL_FEATURE_CONFIG.read_text(encoding="utf-8"))
    features = list(config["features"])
    if len(features) != SPATIAL_N_FEATURES:
        raise AssertionError(
            f"Spatial feature_config has {len(features)} features, expected {SPATIAL_N_FEATURES}"
        )
    if features != list(SPATIAL_FEATURE_COLUMNS):
        raise AssertionError("Spatial feature_config features do not match SPATIAL_FEATURE_COLUMNS")
    if int(config.get("lookback", LOOKBACK)) != LOOKBACK:
        raise AssertionError(f"Spatial feature_config lookback must be {LOOKBACK}")
    if config.get("target") != "Vehicles":
        raise AssertionError("Spatial target must remain Vehicles")
    return features


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
    rows = [{"Model": "Spatial_GRU", "Junction": "Overall", **regression_metrics(y_true, y_pred)}]
    for junction in sorted(meta["Junction"].unique()):
        mask = meta["Junction"].to_numpy() == junction
        rows.append(
            {
                "Model": "Spatial_GRU",
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
    pred_frame: pd.DataFrame,
    spatial_scaler_hashes_before: dict[str, str],
    spatial_scaler_hashes_after: dict[str, str],
    baseline_hashes_before: dict[str, str],
    baseline_hashes_after: dict[str, str],
    used_test_in_fit: bool,
    used_val_as_train_target: bool,
    n_features: int,
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
            len(y_true) == len(y_pred) == len(meta) == len(pred_frame),
            f"actual={len(y_true)} predicted={len(y_pred)} meta={len(meta)} pred_rows={len(pred_frame)}",
        )
    )
    dt_ok = np.array_equal(
        pd.to_datetime(pred_frame["DateTime"]).to_numpy(),
        pd.to_datetime(meta["DateTime"]).to_numpy(),
    )
    j_ok = np.array_equal(
        pred_frame["Junction"].astype(int).to_numpy(),
        meta["Junction"].astype(int).to_numpy(),
    )
    checks.append(
        (
            "datetime_alignment",
            bool(dt_ok and pred_frame["DateTime"].notna().all()),
            "prediction DateTime matches test sequence metadata",
        )
    )
    checks.append(
        (
            "junction_alignment",
            bool(j_ok)
            and pred_frame["Junction"].notna().all()
            and set(int(j) for j in pred_frame["Junction"].unique()) == {1, 2, 3, 4},
            f"junctions={sorted(int(j) for j in pred_frame['Junction'].unique())}",
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
    spatial_ok = spatial_scaler_hashes_before == spatial_scaler_hashes_after
    checks.append(
        (
            "spatial_feature_scaler_unchanged",
            spatial_ok,
            f"feature={spatial_scaler_hashes_after['feature'][:12]}",
        )
    )
    checks.append(
        (
            "spatial_target_scaler_unchanged",
            spatial_ok,
            f"target={spatial_scaler_hashes_after['target'][:12]}",
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
    checks.append(
        (
            "baseline_artifacts_untouched",
            baseline_hashes_before == baseline_hashes_after,
            "baseline GRU model/metrics/predictions and baseline scalers unchanged",
        )
    )
    checks.append(
        (
            "spatial_feature_count",
            n_features == SPATIAL_N_FEATURES,
            f"n_features={n_features}",
        )
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


def _write_training_report(
    path: Path,
    *,
    param_count: int,
    n_features: int,
    lookback: int,
    bundle_shapes: dict[str, Any],
    duration_s: float,
    epochs_completed: int,
    best_epoch: int,
    best_val: float,
    train_losses: list[float],
    val_losses: list[float],
    metric_rows: list[dict[str, Any]],
    checks: list[tuple[str, bool, str]],
) -> None:
    overall = next(row for row in metric_rows if row["Junction"] == "Overall")
    lines = [
        "Spatial GRU training report",
        "",
        "This experiment is separate from the baseline GRU.",
        "No claim is made that Spatial GRU is better than baseline GRU.",
        "Baseline GRU was evaluated on the 2015-2017 full series.",
        "Spatial GRU uses the 2017-01-08 onward four-junction overlap period.",
        "Metrics are not comparable until both models are scored on the same timestamps.",
        "",
        "== Architecture ==",
        f"Input({lookback}, {n_features})",
        f"GRU({GRU_UNITS_1}, return_sequences=True)",
        f"Dropout({DROPOUT})",
        f"GRU({GRU_UNITS_2})",
        f"Dense({DENSE_UNITS}, activation=relu)",
        f"Dense(1, activation={OUTPUT_ACTIVATION})",
        f"parameter_count={param_count}",
        f"feature_count={n_features}",
        f"lookback={lookback}",
        "",
        "== Training configuration ==",
        "optimizer=Adam",
        f"learning_rate={LEARNING_RATE}",
        "loss=MSE",
        f"batch_size={BATCH_SIZE}",
        f"max_epochs={MAX_EPOCHS}",
        f"early_stopping_patience={EARLY_STOPPING_PATIENCE}",
        "restore_best_weights=True",
        f"seed={SEED}",
        f"train_shuffle={TRAIN_SHUFFLE}",
        "target=Vehicles",
        "scalers=models/artifacts/spatial (not refit)",
        "",
        "== Sequence shapes ==",
        f"X_train={bundle_shapes['X_train']} y_train={bundle_shapes['y_train']}",
        f"X_val={bundle_shapes['X_val']} y_val={bundle_shapes['y_val']}",
        f"X_test={bundle_shapes['X_test']} y_test={bundle_shapes['y_test']}",
        f"per_junction={bundle_shapes['per_junction']}",
        "",
        "== Training ==",
        f"duration_seconds={duration_s:.1f}",
        f"duration_minutes={duration_s / 60:.2f}",
        f"epochs_completed={epochs_completed}",
        f"best_epoch={best_epoch}",
        f"best_val_loss={best_val:.6f}",
        f"final_train_loss={train_losses[-1]:.6f}" if train_losses else "final_train_loss=n/a",
        f"final_val_loss={val_losses[-1]:.6f}" if val_losses else "final_val_loss=n/a",
        "",
        "== Overall test (original Vehicles) ==",
        f"RMSE={overall['RMSE']:.4f}",
        f"MAE={overall['MAE']:.4f}",
        f"MAPE={overall['MAPE']:.4f}%",
        f"R2={overall['R2']:.4f}",
        "",
        "== Per-junction test (original Vehicles) ==",
    ]
    for row in metric_rows:
        if row["Junction"] == "Overall":
            continue
        lines.append(
            f"Junction {row['Junction']}: RMSE={row['RMSE']:.4f} MAE={row['MAE']:.4f} "
            f"MAPE={row['MAPE']:.4f}% R2={row['R2']:.4f}"
        )
    lines.extend(["", "== Sanity checks =="])
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        lines.append(f"{status}  {name}: {detail}")
    all_pass = all(passed for _, passed, _ in checks)
    lines.append("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED")
    lines.append("")
    lines.append("Command: python -m ml.training.train_spatial_gru")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    import keras
    from ml.models.gru import build_gru

    set_seeds()
    feature_columns = load_spatial_feature_columns()
    spatial_scaler_hashes_before = {
        "feature": _file_sha256(SPATIAL_FEATURE_SCALER),
        "target": _file_sha256(SPATIAL_TARGET_SCALER),
    }
    baseline_hashes_before = _hash_paths(PROTECTED_BASELINE_PATHS)

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    print(f"KERAS_BACKEND={os.environ.get('KERAS_BACKEND')} keras={keras.__version__} device={device}")
    print(
        "Spatial GRU: 53 features, lookback=168, batch=32, Adam 0.001, "
        f"max_epochs={MAX_EPOCHS}, patience={EARLY_STOPPING_PATIENCE}, "
        f"output={OUTPUT_ACTIVATION}, train_shuffle={TRAIN_SHUFFLE}."
    )
    print(
        "Baseline GRU is not retrained. Baseline artifacts are not overwritten. "
        "Test is never passed to fit()."
    )

    bundle = build_sequences(
        lookback=LOOKBACK,
        processed_dir=SPATIAL_PROCESSED_DIR,
        artifacts_dir=SPATIAL_ARTIFACTS_DIR,
        feature_columns=feature_columns,
        n_features=SPATIAL_N_FEATURES,
    )
    if bundle.n_features != SPATIAL_N_FEATURES:
        raise AssertionError(f"Expected {SPATIAL_N_FEATURES} features, got {bundle.n_features}")
    if bundle.lookback != LOOKBACK:
        raise AssertionError(f"Expected lookback {LOOKBACK}, got {bundle.lookback}")
    print_sequence_report(bundle)
    print(f"X_train={bundle.X_train.shape} X_val={bundle.X_val.shape} X_test={bundle.X_test.shape}")

    model = build_gru(lookback=LOOKBACK, n_features=SPATIAL_N_FEATURES)
    model.summary()
    param_count = int(model.count_params())
    print(f"parameter_count={param_count}")

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

    _, target_scaler = load_scalers(artifacts_dir=SPATIAL_ARTIFACTS_DIR)
    y_pred = predict_vehicles(model, bundle.X_test, target_scaler)
    y_true = inverse_target(bundle.y_test, target_scaler)

    out_dir = SPATIAL_MODEL_DIR
    pred_dir = REPO_ROOT / "results" / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.keras"
    model.save(model_path)

    reloaded = keras.models.load_model(model_path)
    y_pred_reloaded = predict_vehicles(reloaded, bundle.X_test, target_scaler)

    spatial_scaler_hashes_after = {
        "feature": _file_sha256(SPATIAL_FEATURE_SCALER),
        "target": _file_sha256(SPATIAL_TARGET_SCALER),
    }
    baseline_hashes_after = _hash_paths(PROTECTED_BASELINE_PATHS)

    metric_rows = _metrics_by_junction(y_true, y_pred, bundle.meta_test)
    metrics_json = REPO_ROOT / "results" / "spatial_gru_metrics.json"
    metrics_csv = REPO_ROOT / "results" / "spatial_gru_metrics.csv"
    _write_metrics(metric_rows, metrics_json, metrics_csv)

    aligned_frame = pd.DataFrame(
        {
            "DateTime": bundle.meta_test["DateTime"].to_numpy(),
            "Junction": bundle.meta_test["Junction"].astype(int).to_numpy(),
            "Actual": y_true,
            "Predicted": y_pred,
            "Error": y_pred - y_true,
        }
    )
    pred_frame = aligned_frame.sort_values(["Junction", "DateTime"]).reset_index(drop=True)
    pred_path = pred_dir / "spatial_gru_test_predictions.csv"
    pred_frame.to_csv(pred_path, index=False)

    comparison_frame = pred_frame[["DateTime", "Junction", "Actual"]].copy()
    comparison_frame["Spatial_GRU_Predicted"] = pred_frame["Predicted"]
    comparison_path = pred_dir / "spatial_gru_comparison_ready.csv"
    comparison_frame.to_csv(comparison_path, index=False)

    (out_dir / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    config = {
        "mode": "spatial",
        "model": "Spatial_GRU",
        "lookback": LOOKBACK,
        "horizon_steps": HORIZON_STEPS,
        "n_features": bundle.n_features,
        "parameter_count": param_count,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "dropout": DROPOUT,
        "gru_units": [GRU_UNITS_1, GRU_UNITS_2],
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
        "scaler_fitted_on": "spatial train (existing artifacts; not refit)",
        "sequence_counts": bundle.counts,
        "test_not_used_in_fit": True,
        "negative_predictions": int((y_pred < 0).sum()),
        "min_predicted": float(np.min(y_pred)),
        "max_predicted": float(np.max(y_pred)),
        "comparison_note": (
            "Spatial test period is 2017-01-08 onward four-junction overlap. "
            "Baseline GRU metrics used a different temporal period and are not compared here."
        ),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    plot_path = REPO_ROOT / "results" / "plots" / "spatial_gru_training_history.png"
    _save_history_plot(hist, plot_path, "Spatial GRU training history")

    checks = run_sanity_checks(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_reloaded=y_pred_reloaded,
        meta=bundle.meta_test.reset_index(drop=True),
        pred_frame=aligned_frame,
        spatial_scaler_hashes_before=spatial_scaler_hashes_before,
        spatial_scaler_hashes_after=spatial_scaler_hashes_after,
        baseline_hashes_before=baseline_hashes_before,
        baseline_hashes_after=baseline_hashes_after,
        used_test_in_fit=False,
        used_val_as_train_target=False,
        n_features=bundle.n_features,
    )
    checks_ok = _print_checks(checks)

    report_path = REPO_ROOT / "results" / "spatial_gru_training_report.txt"
    _write_training_report(
        report_path,
        param_count=param_count,
        n_features=bundle.n_features,
        lookback=bundle.lookback,
        bundle_shapes={
            "X_train": tuple(bundle.X_train.shape),
            "y_train": tuple(bundle.y_train.shape),
            "X_val": tuple(bundle.X_val.shape),
            "y_val": tuple(bundle.y_val.shape),
            "X_test": tuple(bundle.X_test.shape),
            "y_test": tuple(bundle.y_test.shape),
            "per_junction": bundle.counts["per_junction"],
        },
        duration_s=duration_s,
        epochs_completed=epochs_completed,
        best_epoch=best_epoch,
        best_val=best_val,
        train_losses=train_losses,
        val_losses=val_losses,
        metric_rows=metric_rows,
        checks=checks,
    )

    overall = next(row for row in metric_rows if row["Junction"] == "Overall")
    print("== Training ==")
    print(f"duration_seconds={duration_s:.1f} ({duration_s / 60:.2f} min)")
    print(f"epochs_completed={epochs_completed} best_epoch={best_epoch}")
    print(f"best_val_loss={best_val:.6f}")
    print(f"final_train_loss={train_losses[-1]:.6f}" if train_losses else "final_train_loss=n/a")
    print(f"final_val_loss={val_losses[-1]:.6f}" if val_losses else "final_val_loss=n/a")
    print(f"parameter_count={param_count}")
    print("== Overall test (original Vehicles) ==")
    print(format_metrics(overall))
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
            f"Actual {rec.Actual:.2f} | Predicted {rec.Predicted:.2f} | Error {rec.Error:.2f}"
        )
    print("== Artifacts ==")
    print(model_path)
    print(out_dir / "history.json")
    print(out_dir / "config.json")
    print(metrics_csv)
    print(pred_path)
    print(comparison_path)
    print(plot_path)
    print(report_path)
    print("No ranking versus baseline GRU: test periods differ.")

    print("A. Files created: ml/training/train_spatial_gru.py and spatial GRU artifacts listed above")
    print("B. Files modified: ml/training/sequence_generator.py (optional kwargs only; baseline defaults unchanged)")
    print(
        f"C. Model architecture: Input({LOOKBACK}, {SPATIAL_N_FEATURES}) -> "
        f"GRU({GRU_UNITS_1}, return_sequences=True) -> Dropout({DROPOUT}) -> "
        f"GRU({GRU_UNITS_2}) -> Dense({DENSE_UNITS}, ReLU) -> Dense(1, linear)"
    )
    print(f"D. Parameter count: {param_count}")
    print(
        f"E. Sequence shapes: train={bundle.X_train.shape} "
        f"val={bundle.X_val.shape} test={bundle.X_test.shape}"
    )
    print(f"F. Training duration: {duration_s:.1f} s ({duration_s / 60:.2f} min)")
    print(f"G. Best epoch: {best_epoch}")
    print(f"H. Overall metrics: {format_metrics(overall)}")
    print("I. Per-junction metrics:")
    for row in metric_rows:
        if row["Junction"] == "Overall":
            continue
        print(f"   Junction {row['Junction']}: {format_metrics(row)}")
    print("J. Sanity-check results: " + ("ALL CHECKS PASSED" if checks_ok else "SOME CHECKS FAILED"))
    print("K. Command: python -m ml.training.train_spatial_gru")
    print(
        "L. Baseline artifacts untouched: "
        + ("YES" if baseline_hashes_before == baseline_hashes_after else "NO")
    )
    return 0 if checks_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
