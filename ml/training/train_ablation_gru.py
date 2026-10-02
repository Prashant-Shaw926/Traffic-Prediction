"""Train network and cross-junction ablation GRUs. Does not retrain baseline or full Spatial GRU."""

from __future__ import annotations

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
from ml.features.ablation_features import EXPERIMENTS
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
from ml.training.sequence_generator import ARTIFACTS_DIR, build_sequences, load_scalers, print_sequence_report

SPATIAL_KEYS_PATH = REPO_ROOT / "results" / "predictions" / "spatial_gru_comparison_ready.csv"
FROZEN_PATHS = (
    REPO_ROOT / "models" / "trained" / "gru" / "full" / "model.keras",
    ARTIFACTS_DIR / "feature_scaler.joblib",
    ARTIFACTS_DIR / "target_scaler.joblib",
    REPO_ROOT / "results" / "gru_metrics.csv",
    REPO_ROOT / "results" / "predictions" / "gru_test_predictions.csv",
    REPO_ROOT / "models" / "trained" / "gru" / "spatial" / "model.keras",
    REPO_ROOT / "models" / "artifacts" / "spatial" / "feature_scaler.joblib",
    REPO_ROOT / "models" / "artifacts" / "spatial" / "target_scaler.joblib",
    REPO_ROOT / "results" / "spatial_gru_metrics.csv",
    REPO_ROOT / "results" / "predictions" / "spatial_gru_test_predictions.csv",
    SPATIAL_KEYS_PATH,
    REPO_ROOT / "results" / "model_comparison.csv",
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


def _hash_existing(paths: tuple[Path, ...]) -> dict[str, str]:
    return {str(path): _file_sha256(path) for path in paths if path.is_file()}


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


def train_one(key: str, spec: dict[str, Any], spatial_keys: pd.DataFrame) -> dict[str, Any]:
    import keras
    from ml.models.gru import build_gru

    features = list(spec["features"])
    n_features = int(spec["n_features"])
    processed_dir = REPO_ROOT / "data" / "processed" / "ablation" / key
    artifacts_dir = REPO_ROOT / "models" / "artifacts" / "ablation" / key
    model_dir = REPO_ROOT / "models" / "trained" / "gru" / "ablation" / key
    pred_path = REPO_ROOT / "results" / "ablation" / "predictions" / f"{key}_gru_test_predictions.csv"

    scaler_before = {
        "feature": _file_sha256(artifacts_dir / "feature_scaler.joblib"),
        "target": _file_sha256(artifacts_dir / "target_scaler.joblib"),
    }

    set_seeds()
    bundle = build_sequences(
        lookback=LOOKBACK,
        processed_dir=processed_dir,
        artifacts_dir=artifacts_dir,
        feature_columns=features,
        n_features=n_features,
    )
    print_sequence_report(bundle)
    expected = (11024, LOOKBACK, n_features)
    if bundle.X_train.shape != expected:
        raise AssertionError(f"{key}: expected X_train {expected}, got {bundle.X_train.shape}")
    if bundle.X_test.shape != (2508, LOOKBACK, n_features):
        raise AssertionError(f"{key}: expected X_test (2508, {LOOKBACK}, {n_features}), got {bundle.X_test.shape}")

    model = build_gru(lookback=LOOKBACK, n_features=n_features)
    model.summary()
    param_count = int(model.count_params())

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

    _, target_scaler = load_scalers(artifacts_dir=artifacts_dir)
    y_pred = predict_vehicles(model, bundle.X_test, target_scaler)
    y_true = inverse_target(bundle.y_test, target_scaler)

    pred_frame = pd.DataFrame(
        {
            "DateTime": pd.to_datetime(bundle.meta_test["DateTime"]),
            "Junction": bundle.meta_test["Junction"].astype(int),
            "Actual": y_true,
            "Predicted": y_pred,
            "Error": y_pred - y_true,
        }
    ).sort_values(["Junction", "DateTime"]).reset_index(drop=True)
    pred_keys = pred_frame[["DateTime", "Junction"]].copy()
    if not pred_keys.equals(spatial_keys):
        raise AssertionError(f"{key}: test prediction keys do not match spatial keys")

    model_dir.mkdir(parents=True, exist_ok=True)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.keras"
    model.save(model_path)
    reloaded = keras.models.load_model(model_path)
    y_pred_reloaded = predict_vehicles(reloaded, bundle.X_test, target_scaler)
    max_reload_diff = float(np.max(np.abs(y_pred - y_pred_reloaded))) if len(y_pred) else 0.0

    scaler_after = {
        "feature": _file_sha256(artifacts_dir / "feature_scaler.joblib"),
        "target": _file_sha256(artifacts_dir / "target_scaler.joblib"),
    }
    if scaler_before != scaler_after:
        raise AssertionError(f"{key}: ablation scalers were modified during training")

    pred_frame.to_csv(pred_path, index=False)
    (model_dir / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    overall = regression_metrics(y_true, y_pred)
    config = {
        "mode": "ablation",
        "experiment": spec["name"],
        "feature_group": spec["feature_group"],
        "model": "GRU",
        "lookback": LOOKBACK,
        "horizon_steps": HORIZON_STEPS,
        "n_features": n_features,
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
        "best_val_loss": best_val,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "restore_best_weights": True,
        "train_shuffle": TRAIN_SHUFFLE,
        "seed": SEED,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_duration_seconds": duration_s,
        "scaler_fitted_on": "ablation train only (not refit)",
        "sequence_counts": bundle.counts,
        "test_not_used_in_fit": True,
        "overall_metrics_original_units": overall,
        "reload_max_abs_diff": max_reload_diff,
        "finite_predictions": bool(np.isfinite(y_pred).all()),
    }
    (model_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    plot_path = REPO_ROOT / "results" / "ablation" / "plots" / f"{key}_gru_training_history.png"
    _save_history_plot(hist, plot_path, f"{spec['name']} GRU training history")

    print(f"== {spec['name']} ==")
    print(f"n_features={n_features} params={param_count}")
    print(f"duration_minutes={duration_s / 60:.2f} epochs={epochs_completed} best_epoch={best_epoch}")
    print(format_metrics(overall))
    print(pred_path)
    print(model_path)
    return {
        "experiment": spec["name"],
        "n_features": n_features,
        "param_count": param_count,
        "duration_s": duration_s,
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "X_train": tuple(bundle.X_train.shape),
        "X_val": tuple(bundle.X_val.shape),
        "X_test": tuple(bundle.X_test.shape),
        "metrics": overall,
        "pred_path": str(pred_path),
        "model_path": str(model_path),
        "nan_inf": bool(not np.isfinite(y_pred).all()),
    }


def main() -> int:
    import keras

    hashes_before = _hash_existing(FROZEN_PATHS)
    spatial_keys = pd.read_csv(SPATIAL_KEYS_PATH, parse_dates=["DateTime"])
    spatial_keys["Junction"] = spatial_keys["Junction"].astype(int)
    spatial_keys = spatial_keys[["DateTime", "Junction"]].sort_values(
        ["Junction", "DateTime"]
    ).reset_index(drop=True)

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    print(f"KERAS_BACKEND={os.environ.get('KERAS_BACKEND')} keras={keras.__version__} device={device}")
    print("Training only Network and Cross_Junction ablation GRUs. Baseline and Spatial GRU are frozen.")

    results = []
    for key, spec in EXPERIMENTS.items():
        model_path = REPO_ROOT / "models" / "trained" / "gru" / "ablation" / key / "model.keras"
        pred_path = REPO_ROOT / "results" / "ablation" / "predictions" / f"{key}_gru_test_predictions.csv"
        if model_path.is_file() and pred_path.is_file():
            print(f"Skipping {key}: existing model and predictions found ({model_path})")
            continue
        results.append(train_one(key, spec, spatial_keys))

    hashes_after = _hash_existing(FROZEN_PATHS)
    if hashes_before != hashes_after:
        raise AssertionError("Frozen baseline/spatial artifacts changed during ablation training")

    print("== Ablation training complete ==")
    for row in results:
        print(
            f"{row['experiment']}: n_features={row['n_features']} "
            f"X_test={row['X_test']} {format_metrics(row['metrics'])}"
        )
    print("Frozen artifacts unchanged: PASS")
    print("fit() was not called on baseline or Spatial GRU.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
