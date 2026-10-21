"""STL decomposition and decomposition-based forecasting.

Many notebooks (``Multistep_MLP_with_STL_Decomposition``, ``Prophet_with_STL_...``)
forecast the seasonally adjusted series with a model and add back the seasonal
component. :class:`STLForecaster` packages that recipe around any forecaster.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from poliseye.data import SLOTS_PER_DAY


def stl_decompose(series: pd.Series, period: int = SLOTS_PER_DAY, robust: bool = True):
    """Return a frame with ``observed, trend, seasonal, resid`` columns."""
    res = STL(np.asarray(series, dtype=float), period=period, robust=robust).fit()
    return pd.DataFrame(
        {
            "observed": res.observed,
            "trend": res.trend,
            "seasonal": res.seasonal,
            "resid": res.resid,
        },
        index=series.index if isinstance(series, pd.Series) else None,
    )


class STLForecaster:
    """Forecast ``trend + resid`` with ``base`` and add a seasonal-naive season back."""

    def __init__(self, base, period: int = SLOTS_PER_DAY, robust: bool = True):
        self.base, self.period, self.robust = base, period, robust

    def fit(self, history: pd.Series) -> STLForecaster:
        comp = stl_decompose(pd.Series(np.asarray(history, dtype=float)), self.period, self.robust)
        self.season_ = comp["seasonal"].to_numpy()[-self.period :]
        adjusted = comp["observed"] - comp["seasonal"]
        if isinstance(history, pd.Series):
            adjusted.index = history.index
        self.base.fit(adjusted)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        season = np.tile(self.season_, int(np.ceil(horizon / self.period)))[:horizon]
        return self.base.predict(horizon) + season
