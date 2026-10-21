"""Turn time series into supervised learning samples.

``split_sequence`` and ``split_sequences`` are the helpers repeated in more than 70
notebook cells of ``notebooks/DeepLearningModels`` and ``notebooks/Confronto``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_sequence(
    sequence: np.ndarray | pd.Series, n_steps_in: int, n_steps_out: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over a univariate series.

    Returns ``X`` of shape ``(n_samples, n_steps_in)`` and ``y`` of shape
    ``(n_samples, n_steps_out)``.
    """
    values = np.asarray(sequence, dtype=float)
    if values.ndim != 1:
        raise ValueError("split_sequence expects a 1-D series; use split_sequences")
    n = len(values) - n_steps_in - n_steps_out + 1
    if n <= 0:
        return np.empty((0, n_steps_in)), np.empty((0, n_steps_out))
    X = np.lib.stride_tricks.sliding_window_view(values, n_steps_in)[:n]
    y = np.lib.stride_tricks.sliding_window_view(values[n_steps_in:], n_steps_out)[:n]
    return X.copy(), y.copy()


def split_sequences(
    sequences: np.ndarray | pd.DataFrame, n_steps_in: int, n_steps_out: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over a multivariate series (rows are time, columns are cells).

    Returns ``X`` of shape ``(n_samples, n_steps_in, n_features)`` and ``y`` of shape
    ``(n_samples, n_steps_out, n_features)``.
    """
    values = np.asarray(sequences, dtype=float)
    if values.ndim != 2:
        raise ValueError("split_sequences expects a 2-D array (time x features)")
    n = len(values) - n_steps_in - n_steps_out + 1
    k = values.shape[1]
    if n <= 0:
        return np.empty((0, n_steps_in, k)), np.empty((0, n_steps_out, k))
    X = np.stack([values[i : i + n_steps_in] for i in range(n)])
    y = np.stack([values[i + n_steps_in : i + n_steps_in + n_steps_out] for i in range(n)])
    return X, y


def lag_features(series: pd.Series, lags: list[int]) -> pd.DataFrame:
    """Lagged copies of a series, e.g. ``[1, 96, 672]`` for 15 min, 1 day, 1 week."""
    return pd.DataFrame({f"lag_{lag}": series.shift(lag) for lag in lags}, index=series.index)
