"""Statistical forecasters and grid search (ARIMA, SARIMA, exponential smoothing).

Modernised versions of ``notebooks/legacy_scripts/BestARIMA.py``, ``BestSARIMA.py`` and
``BestETS.py``: they use the current ``statsmodels`` API and the shared forecaster
interface, so a grid search is just a :func:`poliseye.backtest.rolling_origin` call per
configuration.
"""

from __future__ import annotations

import itertools
import warnings
from collections.abc import Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from poliseye.data import SLOTS_PER_DAY


class ARIMAForecaster:
    """(S)ARIMA via ``statsmodels``; pass ``seasonal_order`` for SARIMA."""

    def __init__(
        self,
        order: tuple[int, int, int] = (2, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: str | None = None,
    ):
        self.order, self.seasonal_order, self.trend = order, seasonal_order, trend

    def fit(self, history: pd.Series) -> ARIMAForecaster:
        values = np.asarray(history, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result_ = ARIMA(
                values, order=self.order, seasonal_order=self.seasonal_order, trend=self.trend
            ).fit()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.asarray(self.result_.forecast(horizon), dtype=float)


class ETSForecaster:
    """Holt-Winters exponential smoothing with a daily season by default."""

    def __init__(
        self,
        trend: str | None = None,
        seasonal: str | None = "add",
        seasonal_periods: int = SLOTS_PER_DAY,
        damped_trend: bool = False,
    ):
        self.trend, self.seasonal = trend, seasonal
        self.seasonal_periods, self.damped_trend = seasonal_periods, damped_trend

    def fit(self, history: pd.Series) -> ETSForecaster:
        values = np.asarray(history, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result_ = ExponentialSmoothing(
                values,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods if self.seasonal else None,
                damped_trend=self.damped_trend if self.trend else False,
            ).fit()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.asarray(self.result_.forecast(horizon), dtype=float)


def arima_grid(
    p: Iterable[int] = (0, 1, 2, 4), d: Iterable[int] = (0, 1), q: Iterable[int] = (0, 1, 2)
) -> list[dict]:
    """Parameter grid for :func:`grid_search` with :class:`ARIMAForecaster`."""
    return [{"order": o} for o in itertools.product(p, d, q)]


def grid_search(
    series: pd.Series,
    factory,
    grid: list[dict],
    horizon: int = 1,
    n_origins: int = 8,
    metric: str = "RMSE",
) -> pd.DataFrame:
    """Score every configuration with rolling-origin evaluation, best first.

    Configurations that fail to fit are skipped, as in the original scripts.
    """
    from poliseye.backtest import rolling_origin

    rows = []
    for params in grid:
        try:
            scores = rolling_origin(series, lambda p=params: factory(**p), horizon, n_origins)
        except Exception:  # noqa: BLE001 - mirror the notebooks: skip configs that fail
            continue
        rows.append({"params": params, **scores.mean(numeric_only=True).to_dict()})
    if not rows:
        raise RuntimeError("no configuration could be fitted")
    return pd.DataFrame(rows).sort_values(metric, ignore_index=True)
