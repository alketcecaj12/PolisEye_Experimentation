"""Optional neural forecasters (MLP, CNN, LSTM) used in the comparison study.

Requires the ``deep`` extra: ``pip install "poliseye[deep]"``. The architectures
follow ``notebooks/DeepLearningModels``: a direct multi-output network that maps
``n_steps_in`` past values to ``n_steps_out`` future values of one cell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from poliseye.windows import split_sequence


def _keras():
    try:
        from tensorflow import keras
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "poliseye.deep needs TensorFlow; install it with pip install 'poliseye[deep]'"
        ) from exc
    return keras


def build_model(kind: str, n_steps_in: int, n_steps_out: int, units: int = 100):
    """Build an uncompiled MLP, CNN or LSTM with ``n_steps_out`` outputs."""
    keras = _keras()
    layers = keras.layers
    if kind == "mlp":
        body = [layers.Input((n_steps_in,)), layers.Dense(units, activation="relu")]
    elif kind == "cnn":
        body = [
            layers.Input((n_steps_in, 1)),
            layers.Conv1D(64, 2, activation="relu"),
            layers.MaxPooling1D(2),
            layers.Flatten(),
            layers.Dense(50, activation="relu"),
        ]
    elif kind == "lstm":
        body = [layers.Input((n_steps_in, 1)), layers.LSTM(units, activation="relu")]
    else:
        raise ValueError("kind must be 'mlp', 'cnn' or 'lstm'")
    return keras.Sequential([*body, layers.Dense(n_steps_out)])


class NeuralForecaster:
    """Shared-interface wrapper so neural models run in :mod:`poliseye.backtest`."""

    def __init__(
        self,
        kind: str = "mlp",
        n_steps_in: int = 96,
        n_steps_out: int = 16,
        epochs: int = 50,
        patience: int = 5,
        verbose: int = 0,
    ):
        self.kind, self.n_in, self.n_out = kind, n_steps_in, n_steps_out
        self.epochs, self.patience, self.verbose = epochs, patience, verbose

    def fit(self, history: pd.Series) -> NeuralForecaster:
        keras = _keras()
        values = np.asarray(history, dtype=float)
        self.scale_ = float(values.max()) or 1.0
        X, y = split_sequence(values / self.scale_, self.n_in, self.n_out)
        if self.kind != "mlp":
            X = X[..., None]
        self.model_ = build_model(self.kind, self.n_in, self.n_out)
        self.model_.compile(optimizer="adam", loss="mse")
        stop = keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)
        self.model_.fit(
            X, y, epochs=self.epochs, validation_split=0.1, callbacks=[stop], verbose=self.verbose
        )
        self.last_ = values[-self.n_in :] / self.scale_
        return self

    def predict(self, horizon: int) -> np.ndarray:
        x = self.last_[None, :] if self.kind == "mlp" else self.last_[None, :, None]
        out = self.model_.predict(x, verbose=0)[0] * self.scale_
        if horizon > self.n_out:
            raise ValueError(f"model was trained for horizon <= {self.n_out}")
        return out[:horizon]
