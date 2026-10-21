import datetime as dt

import pandas as pd

from poliseye.timefeatures import calendar_features, easter_sunday, italian_public_holidays


def test_easter():
    assert easter_sunday(2017) == dt.date(2017, 4, 16)
    assert easter_sunday(2024) == dt.date(2024, 3, 31)
    assert easter_sunday(2026) == dt.date(2026, 4, 5)


def test_italian_holidays_match_notebook_2017_list():
    hol = italian_public_holidays(2017)
    for d in ["2017-04-16", "2017-04-17", "2017-04-25", "2017-05-01", "2017-06-02", "2017-08-15"]:
        assert dt.date.fromisoformat(d) in hol


def test_calendar_features():
    idx = pd.date_range("2017-04-24 08:15", periods=3, freq="1D")  # Mon, Tue (25 Apr), Wed
    cal = calendar_features(idx)
    assert cal["slot"].tolist() == [33, 33, 33]
    assert cal["is_holiday"].tolist() == [0, 1, 0]
    assert cal["is_business_day"].tolist() == [1, 0, 1]


def test_custom_holidays():
    idx = pd.date_range("2026-06-05", periods=1, freq="1D")  # Danish Constitution Day
    assert calendar_features(idx, holidays=[dt.date(2026, 6, 5)])["is_holiday"].item() == 1
