"""Rolling-origin (walk-forward) evaluation and model comparison."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from poliseye.metrics import evaluate


def rolling_origin(
    series: pd.Series,
    factory: Callable[[], object],
    horizon: int = 1,
    n_origins: int = 10,
    step: int | None = None,
    min_train: int | None = None,
) -> pd.DataFrame:
    """Refit a fresh model at successive forecast origins at the end of ``series``.

    Each origin trains on everything before it and forecasts ``horizon`` steps.
    Returns one row of error measures per origin.
    """
    n = len(series)
    step = step or horizon
    last_origin = n - horizon
    origins = [last_origin - i * step for i in range(n_origins)][::-1]
    min_train = min_train or max(2, n // 3)
    origins = [o for o in origins if o >= min_train]
    if not origins:
        raise ValueError("series too short for the requested evaluation")
    rows = []
    for origin in origins:
        model = factory().fit(series.iloc[:origin])
        pred = model.predict(horizon)
        actual = series.iloc[origin : origin + horizon].to_numpy(dtype=float)
        rows.append({"origin": series.index[origin], **evaluate(actual, pred)})
    return pd.DataFrame(rows)


def compare_models(
    wide: pd.DataFrame,
    models: Mapping[str, Callable[[], object]],
    horizon: int = 1,
    n_origins: int = 10,
    cells: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate several models on several cells; returns one row per (model, cell)."""
    cells = list(wide.columns) if cells is None else cells
    rows = []
    for cell in cells:
        series = wide[cell].dropna()
        for name, factory in models.items():
            scores = rolling_origin(series, factory, horizon, n_origins)
            rows.append(
                {"model": name, "cell": cell, **scores.drop(columns="origin").mean().to_dict()}
            )
    return pd.DataFrame(rows)


def leaderboard(results: pd.DataFrame, metric: str = "MAE") -> pd.DataFrame:
    """Average across cells and rank models (lower is better)."""
    board = results.groupby("model").mean(numeric_only=True).sort_values(metric)
    board["rank"] = np.arange(1, len(board) + 1)
    return board
