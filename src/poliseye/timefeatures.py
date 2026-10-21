"""Calendar features used by the forecasting notebooks.

Presence and travel demand depend strongly on the type of day. The original EDA
notebooks hard-coded the Italian holidays of 2017; :func:`italian_public_holidays`
computes them for any year, and :func:`calendar_features` accepts any holiday set so
the same code works for, e.g., Danish calendars.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import pandas as pd


def easter_sunday(year: int) -> dt.date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


def italian_public_holidays(year: int) -> set[dt.date]:
    """National public holidays in Italy, including Easter Sunday and Monday."""
    easter = easter_sunday(year)
    fixed = [(1, 1), (1, 6), (4, 25), (5, 1), (6, 2), (8, 15), (11, 1), (12, 8), (12, 25), (12, 26)]
    days = {dt.date(year, m, d) for m, d in fixed}
    days |= {easter, easter + dt.timedelta(days=1)}
    return days


def calendar_features(
    index: pd.DatetimeIndex, holidays: Iterable[dt.date] | None = None
) -> pd.DataFrame:
    """Return time-of-day, day-of-week, business-day and holiday features."""
    index = pd.DatetimeIndex(index)
    if holidays is None:
        holidays = set().union(*(italian_public_holidays(y) for y in set(index.year)))
    holiday_set = {pd.Timestamp(d).date() for d in holidays}
    dates = index.date
    is_holiday = pd.Series([d in holiday_set for d in dates], index=index)
    weekend = index.dayofweek >= 5
    return pd.DataFrame(
        {
            "hour": index.hour,
            "slot": index.hour * 4 + index.minute // 15,
            "dayofweek": index.dayofweek,
            "is_weekend": weekend.astype(int),
            "is_holiday": is_holiday.astype(int).to_numpy(),
            "is_business_day": ((~weekend) & (~is_holiday.to_numpy())).astype(int),
        },
        index=index,
    )
