"""Loading, cleaning and reshaping aggregated mobile phone presence data.

The POLIS-EYE data count the mobile phones observed in each cell of a regular grid
(about 150 m per side) every 15 minutes. Two layouts appear in the notebooks:

* **raw** files without header: ``timestamp, cell_num, nr_people`` where the timestamp
  looks like ``20170416_0000`` (``YYYYMMDD_HHMM``);
* **processed** files: ``date, cell_num, nr_people, hours, minutes``.

Both are normalised here to a *long* frame with columns ``timestamp, cell, count`` and
can be pivoted to a *wide* frame (one column per cell, one row per time slot), which
is the shape every model in the package works on.
"""

from __future__ import annotations

from collections.abc import Sequence
from os import PathLike

import numpy as np
import pandas as pd

#: Value used by the operator to flag a missing measurement.
MISSING_SENTINEL = 65535

#: Sampling interval of the dataset.
FREQ = "15min"

#: Number of 15-minute slots in a day and in a week.
SLOTS_PER_DAY = 96
SLOTS_PER_WEEK = 7 * SLOTS_PER_DAY


def parse_raw_timestamp(values: Sequence[str] | pd.Series) -> pd.DatetimeIndex:
    """Parse ``YYYYMMDD_HHMM`` strings used by the operator's raw export."""
    return pd.DatetimeIndex(
        pd.to_datetime(pd.Series(values).astype(str).str[:13], format="%Y%m%d_%H%M")
    )


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    if {"date", "hours", "minutes"}.issubset(df.columns):
        ts = (
            pd.to_datetime(df["date"])
            + pd.to_timedelta(df["hours"].astype(float), unit="h")
            + pd.to_timedelta(df["minutes"].astype(float), unit="m")
        )
    elif "date" in df.columns and df["date"].astype(str).str.match(r"^\d{8}_\d{4}").all():
        ts = parse_raw_timestamp(df["date"])
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError(
            "Unrecognised layout: expected 'date, hours, minutes', a raw "
            "'YYYYMMDD_HHMM' date, or a 'timestamp' column."
        )
    out = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(ts),
            "cell": df["cell_num"].astype(str).to_numpy(),
            "count": pd.to_numeric(df["nr_people"], errors="coerce").to_numpy(),
        }
    )
    return out.sort_values(["cell", "timestamp"], ignore_index=True)


def load_presence(path: str | PathLike[str], raw: bool | None = None) -> pd.DataFrame:
    """Read a presence file in either layout and return a tidy long frame.

    Parameters
    ----------
    path:
        CSV file.
    raw:
        ``True`` for the headerless operator export, ``False`` for the processed
        layout. ``None`` sniffs the first line.
    """
    if raw is None:
        with open(path, encoding="utf-8") as fh:
            first = fh.readline()
        raw = "cell_num" not in first and "timestamp" not in first
    if raw:
        df = pd.read_csv(path, header=None, names=["date", "cell_num", "nr_people"])
    else:
        df = pd.read_csv(path)
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    return _normalise(df)


def to_wide(long: pd.DataFrame, freq: str = FREQ) -> pd.DataFrame:
    """Pivot a long frame to ``time x cell`` and reindex on a regular time grid."""
    wide = long.pivot_table(index="timestamp", columns="cell", values="count", aggfunc="mean")
    full = pd.date_range(wide.index.min(), wide.index.max(), freq=freq)
    wide = wide.reindex(full)
    wide.index.name = "timestamp"
    wide.columns.name = "cell"
    return wide


def clean_presence(
    wide: pd.DataFrame,
    sentinel: float | None = MISSING_SENTINEL,
    cap: float | None = None,
) -> pd.DataFrame:
    """Replace sentinel values, fill gaps per cell and optionally cap outliers.

    This is the procedure of ``notebooks/EDA/FixData.ipynb``: forward fill then
    backward fill each cell's series, and clip implausible spikes (the notebook used
    a cap of 4500 for Modena).
    """
    out = wide.astype(float).copy()
    if sentinel is not None:
        out = out.mask(out >= sentinel)
    out = out.ffill().bfill()
    if cap is not None:
        out = out.clip(upper=cap)
    return out


def synthetic_presence(
    n_days: int = 28,
    cells: Sequence[str] | None = None,
    start: str = "2017-04-17",
    seed: int = 0,
) -> pd.DataFrame:
    """Generate a realistic wide presence frame for demos and tests.

    Each cell follows one of four archetypal daily rhythms observed in the POLIS-EYE
    grids: a *workplace* cell (weekday daytime peak), a *residential* cell (evening and
    night presence), a *leisure* cell (weekend afternoon peak) and a *transport hub*
    cell (morning and evening peaks, as around Bologna airport).
    """
    rng = np.random.default_rng(seed)
    if cells is None:
        cells = ["486-1258", "488-1264", "490-1254", "495-1266"]
    index = pd.date_range(start, periods=n_days * SLOTS_PER_DAY, freq=FREQ, name="timestamp")
    hour = index.hour.to_numpy() + index.minute.to_numpy() / 60
    weekend = np.asarray(index.dayofweek >= 5)

    def bump(centre: float, width: float) -> np.ndarray:
        return np.exp(-0.5 * ((hour - centre) / width) ** 2)

    archetypes = {
        "workplace": 40 + 260 * bump(12.5, 3.0) * np.where(weekend, 0.25, 1.0),
        "residential": 180
        - 90 * bump(12.0, 3.5) * np.where(weekend, 0.4, 1.0)
        + 40 * bump(20.5, 1.5),
        "leisure": 30 + 120 * bump(16.0, 2.5) * np.where(weekend, 2.2, 0.6),
        "hub": 60 + 200 * bump(8.0, 1.2) + 180 * bump(18.0, 1.5),
    }
    names = list(archetypes)
    data = {}
    for i, cell in enumerate(cells):
        base = archetypes[names[i % len(names)]] * (0.8 + 0.4 * rng.random())
        noise = rng.normal(0, 0.06 * base.mean(), size=base.size)
        data[str(cell)] = np.clip(base + noise, 0, None).round()
    wide = pd.DataFrame(data, index=index)
    wide.columns.name = "cell"
    return wide


def load_cell_errors(path: str | PathLike[str]) -> pd.DataFrame:
    """Read the per-cell error files written by the notebooks (``cell:value`` lines).

    Values may be a scalar or a list (one entry per forecast step); lists are expanded
    to columns ``step_0, step_1, ...``.
    """
    import ast

    rows = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if ":" not in line:
                continue
            cell, value = line.split(":", 1)
            rows[cell.strip()] = ast.literal_eval(value.strip())
    first = next(iter(rows.values()))
    if isinstance(first, (list, tuple)):
        out = pd.DataFrame.from_dict(rows, orient="index")
        out.columns = [f"step_{i}" for i in out.columns]
    else:
        out = pd.Series(rows, name="error", dtype=float).to_frame()
    out.index.name = "cell"
    return out
