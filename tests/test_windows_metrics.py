import numpy as np
import pandas as pd
import pytest

from poliseye.metrics import evaluate, mae, mape, max_ae, rmse
from poliseye.windows import lag_features, split_sequence, split_sequences


def test_split_sequence_matches_notebook_loop():
    seq = np.arange(10)
    X, y = split_sequence(seq, 3, 2)
    assert X.shape == (6, 3) and y.shape == (6, 2)
    assert X[0].tolist() == [0, 1, 2] and y[0].tolist() == [3, 4]
    assert X[-1].tolist() == [5, 6, 7] and y[-1].tolist() == [8, 9]


def test_split_sequence_too_short():
    X, y = split_sequence([1, 2], 3, 1)
    assert X.shape == (0, 3)


def test_split_sequences():
    data = np.arange(20).reshape(10, 2)
    X, y = split_sequences(data, 4, 2)
    assert X.shape == (5, 4, 2) and y.shape == (5, 2, 2)
    assert y[0, 0].tolist() == [8, 9]


def test_lag_features():
    s = pd.Series(range(5))
    lf = lag_features(s, [1, 2])
    assert lf["lag_2"].tolist()[2:] == [0, 1, 2]


def test_metrics():
    y, p = [10, 20, 0], [12, 18, 1]
    assert mae(y, p) == pytest.approx(5 / 3)
    assert rmse(y, p) == pytest.approx(np.sqrt(9 / 3))
    assert mape(y, p) == pytest.approx((20 + 10) / 2)  # zero observation ignored
    assert max_ae(y, p) == 2
    assert set(evaluate(y, p)) == {"MAE", "RMSE", "MAPE", "MaxAE"}


def test_metrics_shape_mismatch():
    with pytest.raises(ValueError):
        mae([1, 2], [1])
