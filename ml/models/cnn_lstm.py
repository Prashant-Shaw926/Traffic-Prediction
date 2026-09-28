"""Paper CNN-LSTM: Conv1D 32 -> MaxPool -> LSTM 128 -> dense 32 ReLU -> 1 linear."""

from __future__ import annotations

from typing import Any

from ml.training.config import (
    CNN_LSTM_UNITS,
    CONV_FILTERS,
    CONV_KERNEL_SIZE,
    DENSE_UNITS,
    LEARNING_RATE,
    N_FEATURES,
    OUTPUT_ACTIVATION,
)


def build_cnn_lstm(
    lookback: int,
    n_features: int = N_FEATURES,
    conv_filters: int = CONV_FILTERS,
    conv_kernel_size: int = CONV_KERNEL_SIZE,
    lstm_units: int = CNN_LSTM_UNITS,
    dense_units: int = DENSE_UNITS,
    learning_rate: float = LEARNING_RATE,
    output_activation: str = OUTPUT_ACTIVATION,
) -> Any:
    """Build and compile the paper CNN-LSTM. Keras must already use the torch backend."""
    import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(lookback, n_features)),
            keras.layers.Conv1D(conv_filters, conv_kernel_size),
            keras.layers.MaxPooling1D(),
            keras.layers.LSTM(lstm_units),
            keras.layers.Dense(dense_units, activation="relu"),
            keras.layers.Dense(1, activation=output_activation),
        ],
        name="cnn_lstm_traffic",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model
