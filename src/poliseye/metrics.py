"""Forecast error measures used to compare methods in the POLIS-EYE experiments."""

from __future__ import annotations

import numpy as np


def _pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true, dtype=float).ravel()
    b = np.asarray(y_pred, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return a, b


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    a, b = _pair(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    a, b = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, in percent, ignoring zero observations."""
    a, b = _pair(y_true, y_pred)
    nz = a != 0
    if not nz.any():
        return float("nan")
    return float(np.mean(np.abs((a[nz] - b[nz]) / a[nz])) * 100)


def max_ae(y_true, y_pred) -> float:
    """Maximum absolute error: the worst-case miss, key for crowd management."""
    a, b = _pair(y_true, y_pred)
    return float(np.max(np.abs(a - b)))


def evaluate(y_true, y_pred) -> dict[str, float]:
    """All error measures at once."""
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "MaxAE": max_ae(y_true, y_pred),
    }
