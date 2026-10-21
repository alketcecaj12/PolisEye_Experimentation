"""Naive forecasters: the yardstick every model has to beat.

All forecasters share a tiny interface, ``fit(history) -> self`` and
``predict(horizon) -> np.ndarray``, so they plug into :mod:`poliseye.backtest`.
They port ``notebooks/StatisticalModels/NaiveMethods``.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from poliseye.data import SLOTS_PER_DAY, SLOTS_PER_WEEK


class Forecaster(Protocol):
    def fit(self, history: pd.Series) -> Forecaster: ...

    def predict(self, horizon: int) -> np.ndarray: ...


class PersistenceForecaster:
    """Repeat the last observation ("valore previsto = valore precedente")."""

    def fit(self, history: pd.Series) -> PersistenceForecaster:
        self.last_ = float(np.asarray(history)[-1])
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.last_)


class SeasonalNaiveForecaster:
    """Repeat the value observed one season earlier (one day by default)."""

    def __init__(self, period: int = SLOTS_PER_DAY):
        self.period = period

    def fit(self, history: pd.Series) -> SeasonalNaiveForecaster:
        values = np.asarray(history, dtype=float)
        if len(values) < self.period:
            raise ValueError("history shorter than one season")
        self.season_ = values[-self.period :]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        reps = int(np.ceil(horizon / self.period))
        return np.tile(self.season_, reps)[:horizon]


class MovingAverageForecaster:
    """Mean (or median) of the last ``window`` observations taken every ``offset`` slots.

    With ``offset=96`` this averages the same time of day over previous days.
    """

    def __init__(self, window: int = 4, offset: int = 1, how: str = "mean"):
        if how not in {"mean", "median"}:
            raise ValueError("how must be 'mean' or 'median'")
        self.window, self.offset, self.how = window, offset, how

    def fit(self, history: pd.Series) -> MovingAverageForecaster:
        self.history_ = np.asarray(history, dtype=float)
        if self.window * self.offset > len(self.history_):
            raise ValueError("configuration reaches beyond the start of the history")
        return self

    def predict(self, horizon: int) -> np.ndarray:
        agg = np.mean if self.how == "mean" else np.median
        h = self.history_
        out = []
        for step in range(1, horizon + 1):
            if self.offset == 1:
                out.append(agg(h[-self.window :]))
            else:
                # value at the same phase, `i` seasons back, for the step being forecast
                idx = [
                    len(h) - i * self.offset + (step - 1) % self.offset
                    for i in range(1, self.window + 1)
                ]
                out.append(agg(h[idx]))
        return np.asarray(out)


class DayHourProfileForecaster:
    """Average presence for the same weekday and 15-minute slot.

    This is the strongest naive baseline in the notebooks
    (``error_data_4_Naive_DayHourAveragePrediction``) and also a direct estimate
    of the typical weekly activity rhythm of an area.
    """

    def __init__(self, how: str = "mean"):
        self.how = how

    def fit(self, history: pd.Series) -> DayHourProfileForecaster:
        if not isinstance(history.index, pd.DatetimeIndex):
            raise TypeError("DayHourProfileForecaster needs a DatetimeIndex")
        key = history.index.dayofweek * SLOTS_PER_DAY + (
            history.index.hour * 4 + history.index.minute // 15
        )
        self.profile_ = history.groupby(key).agg(self.how).reindex(range(SLOTS_PER_WEEK))
        self.profile_ = self.profile_.fillna(history.mean())
        self.freq_ = pd.Timedelta(minutes=15)
        self.next_ = history.index[-1] + self.freq_
        return self

    def predict(self, horizon: int) -> np.ndarray:
        idx = pd.date_range(self.next_, periods=horizon, freq=self.freq_)
        key = idx.dayofweek * SLOTS_PER_DAY + idx.hour * 4 + idx.minute // 15
        return self.profile_.loc[key].to_numpy(dtype=float)
