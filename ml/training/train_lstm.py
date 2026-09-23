"""Train the paper LSTM. Use --smoke for a tiny Junction-1 run only."""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timezone

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np

from ml.data.load_data import REPO_ROOT
from ml.inference.predictor import inverse_target, predict_vehicles
from ml.evaluation.metrics import format_metrics, regression_metrics
from ml.training.config import (
    BATCH_SIZE,
    DENSE_UNITS,
    DROPOUT,
    HORIZON_STEPS,
    LEARNING_RATE,
    LOOKBACK,
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    SEED,
    SMOKE_EARLY_STOPPING_PATIENCE,
    SMOKE_EPOCHS,
    SMOKE_JUNCTION,
    SMOKE_TRAIN_WINDOWS,
    SMOKE_VAL_WINDOWS,
)
from ml.training.sequence_generator import build_sequences, load_scalers, print_sequence_report


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


def _subset(bundle, train_n: int, val_n: int):
    return (
        bundle.X_train[:train_n],
        bundle.y_train[:train_n],
        bundle.meta_train.iloc[:train_n].reset_index(drop=True),
        bundle.X_val[:val_n],
        bundle.y_val[:val_n],
        bundle.meta_val.iloc[:val_n].reset_index(drop=True),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Junction-1, 256/64 windows, 2 epochs. Not a real evaluation.",
    )
    args = parser.parse_args()
    if not args.smoke:
        print("Full LSTM training is not enabled in this slice. Re-run with --smoke.")
        return 1

    set_seeds()
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
    X_train, y_train, meta_train, X_val, y_val, meta_val = _subset(
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
    plot_dir = REPO_ROOT / "results" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model.keras")
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2), encoding="utf-8")
    config = {
        "mode": "smoke",
        "lookback": LOOKBACK,
        "horizon_steps": HORIZON_STEPS,
        "n_features": bundle.n_features,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "dropout": DROPOUT,
        "lstm_units": [LSTM_UNITS_1, LSTM_UNITS_2],
        "dense_units": DENSE_UNITS,
        "epochs_requested": SMOKE_EPOCHS,
        "early_stopping_patience": SMOKE_EARLY_STOPPING_PATIENCE,
        "seed": SEED,
        "junctions": [SMOKE_JUNCTION],
        "smoke_train_windows": SMOKE_TRAIN_WINDOWS,
        "smoke_val_windows": SMOKE_VAL_WINDOWS,
        "keras_backend": os.environ.get("KERAS_BACKEND"),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "scaler_fitted_on": "train (existing artifacts; not refit)",
        "full_sequence_counts": bundle.counts,
        "smoke_metrics_original_units": metrics,
        "not_a_real_evaluation": True,
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 3.5))
        plt.plot(hist.get("loss", []), label="train")
        plt.plot(hist.get("val_loss", []), label="val")
        plt.xlabel("epoch")
        plt.ylabel("MSE (scaled)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "lstm_smoke_loss.png", dpi=120)
        plt.close()
    except Exception as exc:
        print(f"Could not save smoke loss plot: {exc}")

    print(f"Saved smoke artifacts under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
