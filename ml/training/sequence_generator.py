"""Sliding-window sequences grouped by Junction. No shuffle, no future leakage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from ml.data.load_data import REPO_ROOT
from ml.features.feature_pipeline import FEATURE_COLUMNS, TARGET_COLUMN
from ml.training.config import LOOKBACK, N_FEATURES

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
ARTIFACTS_DIR = REPO_ROOT / "models" / "artifacts"


@dataclass
class SequenceBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    meta_train: pd.DataFrame
    X_val: np.ndarray
    y_val: np.ndarray
    meta_val: pd.DataFrame
    X_test: np.ndarray
    y_test: np.ndarray
    meta_test: pd.DataFrame
    counts: dict[str, Any]
    lookback: int
    n_features: int


def load_processed_frame() -> pd.DataFrame:
    """Concat train/val/test so val/test windows can use earlier context."""
    frames = []
    for name in ("train", "validation", "test"):
        path = PROCESSED_DIR / f"{name}.csv"
        part = pd.read_csv(path, parse_dates=["DateTime"])
        if "split" not in part.columns:
            part["split"] = "validation" if name == "validation" else name
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["Junction", "DateTime"]).reset_index(drop=True)


def load_scalers() -> tuple[Any, Any]:
    """Load scalers fitted on training data only. Never refit."""
    feature_scaler = joblib.load(ARTIFACTS_DIR / "feature_scaler.joblib")
    target_scaler = joblib.load(ARTIFACTS_DIR / "target_scaler.joblib")
    return feature_scaler, target_scaler


def verify_split_order(df: pd.DataFrame) -> None:
    """Train targets must end before validation, which must end before test."""
    for junction, group in df.groupby("Junction", sort=True):
        train_end = group.loc[group["split"] == "train", "DateTime"].max()
        val_times = group.loc[group["split"] == "validation", "DateTime"]
        test_start = group.loc[group["split"] == "test", "DateTime"].min()
        if val_times.empty:
            raise AssertionError(f"Junction {junction}: missing validation rows")
        if not (train_end < val_times.min() <= val_times.max() < test_start):
            raise AssertionError(
                f"Junction {junction}: split order failed "
                f"train_end={train_end} val=[{val_times.min()}, {val_times.max()}] "
                f"test_start={test_start}"
            )


def _windows_for_junction(
    group: pd.DataFrame,
    lookback: int,
    feature_scaler: Any,
    target_scaler: Any,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    group = group.sort_values("DateTime").reset_index(drop=True)
    if group["Junction"].nunique() != 1:
        raise AssertionError("Window builder received mixed Junctions")
    if not group["DateTime"].is_monotonic_increasing:
        raise AssertionError("Junction series is not sorted by DateTime")

    features = np.asarray(feature_scaler.transform(group[FEATURE_COLUMNS]))
    target = np.asarray(target_scaler.transform(group[[TARGET_COLUMN]])).ravel()
    if features.shape[1] != N_FEATURES:
        raise AssertionError(
            f"Expected {N_FEATURES} features, got {features.shape[1]}"
        )
    if len(group) < lookback:
        empty_meta = pd.DataFrame(columns=["DateTime", "Junction", "split"])
        return (
            np.empty((0, lookback, N_FEATURES), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            empty_meta,
        )

    # windows[i] = rows [i, i+lookback); target is the last row of the window (T)
    X = sliding_window_view(features, (lookback, N_FEATURES))[:, 0, :, :].copy()
    y = target[lookback - 1 :]
    meta = group.iloc[lookback - 1 :][["DateTime", "Junction", "split"]].reset_index(
        drop=True
    )
    times = group["DateTime"].to_numpy()
    for i in range(len(meta)):
        window_times = times[i : i + lookback]
        t = times[i + lookback - 1]
        if window_times.max() > t:
            raise AssertionError("Sequence contains a timestamp after T")
    X = X.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    return X, y, meta


def _split_bundle(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]]:
    out: dict[str, tuple[np.ndarray, np.ndarray, pd.DataFrame]] = {}
    for name in ("train", "validation", "test"):
        mask = meta["split"].to_numpy() == name
        out[name] = (X[mask], y[mask], meta.loc[mask].reset_index(drop=True))
    return out


def build_sequences(
    lookback: int = LOOKBACK,
    junctions: list[int] | None = None,
) -> SequenceBundle:
    """
    Build (samples, lookback, features) tensors.

    Each sample predicts Vehicles(T) from feature rows T-lookback+1 ... T.
    Features at T include vehicles_lag_1 = Vehicles(T-1), never Vehicles(T).
    Sample split is the split of T. Earlier rows may come from a previous split.
    """
    df = load_processed_frame()
    verify_split_order(df)
    feature_scaler, target_scaler = load_scalers()

    xs_train, ys_train, ms_train = [], [], []
    xs_val, ys_val, ms_val = [], [], []
    xs_test, ys_test, ms_test = [], [], []
    per_junction: dict[str, dict[str, int]] = {}

    selected = sorted(df["Junction"].unique().tolist())
    if junctions is not None:
        selected = [j for j in selected if int(j) in set(int(x) for x in junctions)]

    for junction, group in df.groupby("Junction", sort=True):
        if int(junction) not in set(int(j) for j in selected):
            continue
        X, y, meta = _windows_for_junction(
            group, lookback, feature_scaler, target_scaler
        )
        parts = _split_bundle(X, y, meta)
        per_junction[str(int(junction))] = {
            "train": int(len(parts["train"][1])),
            "validation": int(len(parts["validation"][1])),
            "test": int(len(parts["test"][1])),
        }
        xs_train.append(parts["train"][0])
        ys_train.append(parts["train"][1])
        ms_train.append(parts["train"][2])
        xs_val.append(parts["validation"][0])
        ys_val.append(parts["validation"][1])
        ms_val.append(parts["validation"][2])
        xs_test.append(parts["test"][0])
        ys_test.append(parts["test"][1])
        ms_test.append(parts["test"][2])

    def _stack(
        xs: list[np.ndarray],
        ys: list[np.ndarray],
        ms: list[pd.DataFrame],
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        if not xs:
            empty_x = np.empty((0, lookback, N_FEATURES), dtype=np.float32)
            empty_y = np.empty((0,), dtype=np.float32)
            empty_m = pd.DataFrame(columns=["DateTime", "Junction", "split"])
            return empty_x, empty_y, empty_m
        return np.concatenate(xs), np.concatenate(ys), pd.concat(ms, ignore_index=True)

    X_train, y_train, meta_train = _stack(xs_train, ys_train, ms_train)
    X_val, y_val, meta_val = _stack(xs_val, ys_val, ms_val)
    X_test, y_test, meta_test = _stack(xs_test, ys_test, ms_test)

    counts = {
        "lookback": lookback,
        "n_features": N_FEATURES,
        "per_junction": per_junction,
        "train": int(len(y_train)),
        "validation": int(len(y_val)),
        "test": int(len(y_test)),
        "note": (
            "Windows end at T. Target is Vehicles(T). "
            "Input uses traffic only through T-1 via lag features. "
            "Val/test T may include earlier-split rows as context."
        ),
    }
    return SequenceBundle(
        X_train=X_train,
        y_train=y_train,
        meta_train=meta_train,
        X_val=X_val,
        y_val=y_val,
        meta_val=meta_val,
        X_test=X_test,
        y_test=y_test,
        meta_test=meta_test,
        counts=counts,
        lookback=lookback,
        n_features=N_FEATURES,
    )


def print_sequence_report(bundle: SequenceBundle) -> None:
    print("== Sequence report ==")
    print(f"lookback={bundle.lookback}")
    print(f"n_features={bundle.n_features}")
    print(f"X_train={bundle.X_train.shape} y_train={bundle.y_train.shape}")
    print(f"X_val={bundle.X_val.shape} y_val={bundle.y_val.shape}")
    print(f"X_test={bundle.X_test.shape} y_test={bundle.y_test.shape}")
    print("counts per junction:")
    for junction, counts in bundle.counts["per_junction"].items():
        print(f"  Junction {junction}: {counts}")
    print(bundle.counts["note"])
