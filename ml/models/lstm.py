"""Paper LSTM: 128 -> dropout -> 64 -> dense 32 ReLU -> 1."""

from __future__ import annotations

from typing import Any

from ml.training.config import (
    DENSE_UNITS,
    DROPOUT,
    LEARNING_RATE,
    LSTM_UNITS_1,
    LSTM_UNITS_2,
    N_FEATURES,
    OUTPUT_ACTIVATION,
)


def build_lstm(
    lookback: int,
    n_features: int = N_FEATURES,
    lstm_units_1: int = LSTM_UNITS_1,
    lstm_units_2: int = LSTM_UNITS_2,
    dropout: float = DROPOUT,
    dense_units: int = DENSE_UNITS,
    learning_rate: float = LEARNING_RATE,
    output_activation: str = OUTPUT_ACTIVATION,
) -> Any:
    """Build and compile the configurable LSTM. Keras must already use the torch backend."""
    import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(lookback, n_features)),
            keras.layers.LSTM(lstm_units_1, return_sequences=True),
            keras.layers.Dropout(dropout),
            keras.layers.LSTM(lstm_units_2),
            keras.layers.Dense(dense_units, activation="relu"),
            keras.layers.Dense(1, activation=output_activation),
        ],
        name="lstm_traffic",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model
