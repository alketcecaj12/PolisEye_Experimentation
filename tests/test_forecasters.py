import numpy as np
import pandas as pd
import pytest

from poliseye.backtest import compare_models, leaderboard, rolling_origin
from poliseye.baselines import (
    DayHourProfileForecaster,
    MovingAverageForecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)
from poliseye.decomposition import STLForecaster, stl_decompose
from poliseye.statistical import ARIMAForecaster, ETSForecaster, arima_grid, grid_search


@pytest.fixture
def series(wide):
    return wide["486-1258"]


def test_persistence(series):
    assert PersistenceForecaster().fit(series).predict(3).tolist() == [series.iloc[-1]] * 3


def test_seasonal_naive_repeats_last_day(series):
    pred = SeasonalNaiveForecaster(96).fit(series).predict(100)
    assert np.allclose(pred[:96], series.iloc[-96:])
    assert pred[96] == series.iloc[-96]


def test_moving_average_same_slot(series):
    pred = MovingAverageForecaster(window=2, offset=96).fit(series).predict(1)
    assert pred[0] == pytest.approx(np.mean([series.iloc[-96], series.iloc[-192]]))


def test_profile_beats_persistence_on_daily_rhythm(series):
    p = rolling_origin(series, DayHourProfileForecaster, horizon=8, n_origins=5)["MAE"].mean()
    n = rolling_origin(series, PersistenceForecaster, horizon=8, n_origins=5)["MAE"].mean()
    assert p < n


def test_profile_requires_datetime_index():
    with pytest.raises(TypeError):
        DayHourProfileForecaster().fit(pd.Series([1.0, 2.0]))


def test_rolling_origin_rows(series):
    res = rolling_origin(series, PersistenceForecaster, horizon=4, n_origins=3)
    assert len(res) == 3
    assert res["origin"].is_monotonic_increasing


def test_compare_and_leaderboard(wide):
    res = compare_models(
        wide,
        {"persistence": PersistenceForecaster, "profile": DayHourProfileForecaster},
        horizon=4,
        n_origins=3,
    )
    assert len(res) == 2 * wide.shape[1]
    board = leaderboard(res)
    assert board.index[0] == "profile"


def test_stl(series):
    comp = stl_decompose(series.iloc[: 96 * 7])
    assert np.allclose(comp[["trend", "seasonal", "resid"]].sum(axis=1), comp["observed"])
    pred = STLForecaster(PersistenceForecaster()).fit(series.iloc[: 96 * 7]).predict(10)
    assert pred.shape == (10,)


def test_statistical_forecasters(series):
    short = series.iloc[-96 * 4 :]
    assert ARIMAForecaster((1, 0, 0)).fit(short).predict(5).shape == (5,)
    assert ETSForecaster().fit(short).predict(5).shape == (5,)


def test_grid_search(series):
    short = series.iloc[-96 * 3 :]
    res = grid_search(
        short, ARIMAForecaster, arima_grid(p=(0, 1), d=(0,), q=(0,)), horizon=1, n_origins=2
    )
    assert len(res) == 2
    assert res["RMSE"].is_monotonic_increasing
