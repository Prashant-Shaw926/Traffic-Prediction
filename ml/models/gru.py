"""Paper GRU: 128 -> dropout -> 64 -> dense 32 ReLU -> 1 linear."""

from __future__ import annotations

from typing import Any

from ml.training.config import (
    DENSE_UNITS,
    DROPOUT,
    GRU_UNITS_1,
    GRU_UNITS_2,
    LEARNING_RATE,
    N_FEATURES,
    OUTPUT_ACTIVATION,
)


def build_gru(
    lookback: int,
    n_features: int = N_FEATURES,
    gru_units_1: int = GRU_UNITS_1,
    gru_units_2: int = GRU_UNITS_2,
    dropout: float = DROPOUT,
    dense_units: int = DENSE_UNITS,
    learning_rate: float = LEARNING_RATE,
    output_activation: str = OUTPUT_ACTIVATION,
) -> Any:
    """Build and compile the configurable GRU. Keras must already use the torch backend."""
    import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(lookback, n_features)),
            keras.layers.GRU(gru_units_1, return_sequences=True),
            keras.layers.Dropout(dropout),
            keras.layers.GRU(gru_units_2),
            keras.layers.Dense(dense_units, activation="relu"),
            keras.layers.Dense(1, activation=output_activation),
        ],
        name="gru_traffic",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model
