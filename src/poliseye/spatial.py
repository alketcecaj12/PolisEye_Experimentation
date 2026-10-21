"""Spatio-temporal dependence between grid cells.

Ports ``notebooks/SpatioTemporalCA``: Pearson correlation, time-lagged cross
correlation (TLCC), windowed TLCC and Granger causality. In travel-behaviour terms,
a cell whose presence *leads* another's by some lag hints at a flow between them (for
example an airport terminal leading a nearby car park or rail stop).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def crosscorr(x: pd.Series, y: pd.Series, lag: int = 0, wrap: bool = False) -> float:
    """Pearson correlation between ``x`` and ``y`` shifted by ``lag`` steps."""
    x = pd.Series(np.asarray(x, dtype=float))
    y = pd.Series(np.asarray(y, dtype=float))
    if wrap:
        return float(x.corr(pd.Series(np.roll(y.to_numpy(), lag))))
    return float(x.corr(y.shift(lag)))


def lagged_cross_correlation(x: pd.Series, y: pd.Series, max_lag: int = 8) -> pd.Series:
    """TLCC for lags ``-max_lag..max_lag``. A peak at lag ``k > 0`` means x leads y by k."""
    lags = range(-max_lag, max_lag + 1)
    # corr(x_t, y_{t+k}) peaks at k when y follows x by k steps
    return pd.Series([crosscorr(x, y, -k) for k in lags], index=pd.Index(lags, name="lag"))


def peak_lag(x: pd.Series, y: pd.Series, max_lag: int = 8) -> tuple[int, float]:
    """Lag with the strongest correlation and its value."""
    tlcc = lagged_cross_correlation(x, y, max_lag)
    k = int(tlcc.idxmax())
    return k, float(tlcc.loc[k])


def windowed_lagged_cross_correlation(
    x: pd.Series, y: pd.Series, window: int = 96, max_lag: int = 8
) -> pd.DataFrame:
    """TLCC computed in consecutive non-overlapping windows (rows) per lag (columns)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rows = {}
    for start in range(0, len(x) - window + 1, window):
        xs = pd.Series(x[start : start + window])
        ys = pd.Series(y[start : start + window])
        rows[start] = lagged_cross_correlation(xs, ys, max_lag)
    return pd.DataFrame(rows).T.rename_axis("window_start")


def correlation_matrix(wide: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Cell-by-cell correlation of presence time series."""
    return wide.corr(method=method)


def granger_matrix(wide: pd.DataFrame, maxlag: int = 4) -> pd.DataFrame:
    """Minimum p-value (over lags 1..maxlag) of "column causes row" Granger tests.

    Series are first-differenced to remove the shared daily level.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    diffed = wide.astype(float).diff().dropna()
    cells = list(diffed.columns)
    out = pd.DataFrame(np.nan, index=cells, columns=cells)
    for effect in cells:
        for cause in cells:
            if cause == effect:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = grangercausalitytests(diffed[[effect, cause]].to_numpy(), maxlag=maxlag)
            out.loc[effect, cause] = min(r[0]["ssr_ftest"][1] for r in res.values())
    out.index.name, out.columns.name = "effect", "cause"
    return out
