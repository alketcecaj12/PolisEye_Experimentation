import numpy as np
import pandas as pd

from poliseye.data import (
    MISSING_SENTINEL,
    clean_presence,
    load_presence,
    parse_raw_timestamp,
    to_wide,
)


def test_parse_raw_timestamp():
    ts = parse_raw_timestamp(["20170416_0000", "20170416_1345"])
    assert list(ts) == [pd.Timestamp("2017-04-16 00:00"), pd.Timestamp("2017-04-16 13:45")]


def test_load_processed_layout(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text(
        "date,cell_num,nr_people,hours,minutes\n"
        "2017-04-16,486-1258,42.0,0.0,0.0\n"
        "2017-04-16,486-1258,40.0,0.0,15.0\n"
        "2017-04-16,488-1264,7.0,0.0,0.0\n"
    )
    long = load_presence(f)
    assert list(long.columns) == ["timestamp", "cell", "count"]
    assert long.loc[1, "timestamp"] == pd.Timestamp("2017-04-16 00:15")


def test_load_raw_layout(tmp_path):
    f = tmp_path / "raw"
    f.write_text("20170416_0000,486-1258,42\n20170416_0015,486-1258,65535\n")
    long = load_presence(f)
    assert long["count"].tolist() == [42, MISSING_SENTINEL]


def test_to_wide_regular_grid_and_clean():
    long = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2017-04-16 00:00", "2017-04-16 00:30"] * 2),
            "cell": ["a", "a", "b", "b"],
            "count": [10, 12, MISSING_SENTINEL, 5],
        }
    )
    wide = to_wide(long)
    assert len(wide) == 3  # the 00:15 slot is inserted
    clean = clean_presence(wide, cap=11)
    assert not clean.isna().any().any()
    assert clean["a"].tolist() == [10, 10, 11]
    assert clean["b"].tolist() == [5, 5, 5]


def test_synthetic_presence_shape(wide):
    assert wide.shape == (21 * 96, 4)
    assert (wide.to_numpy() >= 0).all()
    assert pd.infer_freq(wide.index) in {"15min", "15T"}
    assert np.isfinite(wide.to_numpy()).all()


def test_load_cell_errors(tmp_path):
    from poliseye.data import load_cell_errors

    f = tmp_path / "MAPE.csv"
    f.write_text("486-1252:9.28\n486-1253:9.25\n")
    assert load_cell_errors(f)["error"].tolist() == [9.28, 9.25]
    g = tmp_path / "MAE.csv"
    g.write_text("486-1252:[0.1, 1.5, 3.1]\n")
    assert list(load_cell_errors(g).columns) == ["step_0", "step_1", "step_2"]


def test_load_cell_errors_on_notebook_output():
    from pathlib import Path

    from poliseye.data import load_cell_errors

    f = (
        Path(__file__).parents[1]
        / "notebooks/StatisticalModels/NaiveMethods"
        / ("MAPE_error_data_4_Naive_DayHourAveragePrediction.csv")
    )
    if f.exists():
        assert len(load_cell_errors(f)) > 200
