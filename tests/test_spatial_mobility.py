import numpy as np
import pandas as pd
import pytest

from poliseye.mobility import (
    activity_levels,
    anomaly_scores,
    area_typologies,
    daily_profile,
    mobility_indicators,
)
from poliseye.spatial import (
    correlation_matrix,
    crosscorr,
    granger_matrix,
    lagged_cross_correlation,
    peak_lag,
    windowed_lagged_cross_correlation,
)


def _leader_follower(lag=3, n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.Series(rng.normal(size=n)).rolling(3, min_periods=1).mean()
    y = x.shift(lag).bfill() + rng.normal(scale=0.05, size=n)
    return x, y


def test_crosscorr_self():
    x, _ = _leader_follower()
    assert crosscorr(x, x) == pytest.approx(1.0)


def test_peak_lag_recovers_leader():
    x, y = _leader_follower(lag=3)
    k, r = peak_lag(x, y, max_lag=6)
    assert k == 3 and r > 0.9
    assert len(lagged_cross_correlation(x, y, 6)) == 13


def test_windowed_tlcc_shape():
    x, y = _leader_follower(n=384)
    w = windowed_lagged_cross_correlation(x, y, window=96, max_lag=4)
    assert w.shape == (4, 9)


def test_granger_detects_direction():
    x, y = _leader_follower(lag=2, n=500)
    g = granger_matrix(pd.DataFrame({"x": x, "y": y}), maxlag=3)
    assert g.loc["y", "x"] < 0.01  # x Granger-causes y


def test_correlation_matrix(wide):
    c = correlation_matrix(wide)
    assert np.allclose(np.diag(c), 1)


def test_daily_profile(wide):
    prof = daily_profile(wide)
    assert set(prof.index.get_level_values(0)) == {"business", "non_business"}
    assert prof.xs("business").shape == (96, wide.shape[1])


def test_mobility_indicators_capture_archetypes(wide):
    ind = mobility_indicators(wide)
    assert ind.loc["486-1258", "weekday_weekend"] > 1.2  # workplace
    assert ind.loc["490-1254", "weekday_weekend"] < 0.8  # leisure
    assert ind.loc["488-1264", "night_share"] == ind["night_share"].max()  # residential
    assert ind.loc["495-1266", "peak_time"] in {"07:45", "08:00", "08:15"}  # hub


def test_activity_levels(wide):
    lv = activity_levels(wide)
    assert set(lv.cat.categories) == {"low", "medium", "high", "very_high"}
    assert len(lv) == wide.shape[1]


def test_area_typologies_labels(wide):
    typ = area_typologies(wide, n_types=4)
    assert typ.loc["486-1258", "typology"] == "workplace"
    assert typ.loc["488-1264", "typology"] == "residential"
    assert typ.loc["490-1254", "typology"] == "leisure"
    assert typ.loc["495-1266", "typology"] == "transport hub"


def test_anomaly_scores_flag_event(wide):
    w = wide.copy()
    w.iloc[-10, 0] *= 4
    z = anomaly_scores(w, history_days=14)
    assert z.iloc[-10, 0] > 5
    assert abs(np.nanmedian(z.to_numpy())) < 1
