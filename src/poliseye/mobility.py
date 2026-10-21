"""Travel-behaviour indicators from cell-level presence.

Presence counts are the footprint of trips: people arrive in a cell, stay, and leave.
The indicators below summarise *when* and *how strongly* each area is used, which is
what planners need to reason about peak spreading, land-use mix and demand
management:

* :func:`daily_profile` typical presence by time of day, for business and non-business days
* :func:`mobility_indicators` one row per cell: peak time, peak-to-base ratio, AM/PM
  balance, weekday/weekend ratio, night share, variability
* :func:`activity_levels` the quartile classification of
  ``Classifica_celle_per_livello_di_attività_e_profilo.ipynb``
* :func:`area_typologies` clusters cells by weekly rhythm into functional typologies
  (workplace, residential, leisure, transport hub...)
* :func:`anomaly_scores` departures from the typical rhythm, for event detection
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from poliseye.data import SLOTS_PER_DAY, SLOTS_PER_WEEK
from poliseye.timefeatures import calendar_features


def _slot(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.hour * 4 + index.minute // 15).to_numpy()


def daily_profile(wide: pd.DataFrame, holidays=None) -> pd.DataFrame:
    """Mean presence per 15-minute slot, split into business and non-business days.

    Returns a frame indexed by ``(day_type, slot)`` with one column per cell.
    """
    cal = calendar_features(wide.index, holidays)
    day_type = np.where(cal["is_business_day"] == 1, "business", "non_business")
    return wide.groupby([day_type, _slot(wide.index)]).mean().rename_axis(["day_type", "slot"])


def mobility_indicators(wide: pd.DataFrame, holidays=None) -> pd.DataFrame:
    """Summary indicators of temporal use for every cell.

    Columns
    -------
    mean_presence      average count
    peak_time          time of day of the business-day peak (``HH:MM``)
    peak_to_base       business-day peak / night minimum: how "peaky" the demand is
    am_pm_ratio        mean 07-10 presence / mean 16-19 presence on business days;
                       < 1 for evening attractors, ~1 for hubs with both peaks
    weekday_weekend    business-day mean / non-business-day mean; > 1 for workplaces
    night_share        share of daily presence between 00:00 and 06:00 (residential signal)
    cv                 coefficient of variation of the whole series
    """
    prof = daily_profile(wide, holidays)
    cal = calendar_features(wide.index, holidays)
    biz = cal["is_business_day"].to_numpy() == 1
    hours = wide.index.hour
    rows = {}
    for cell in wide.columns:
        s = wide[cell].astype(float)
        day_types = prof.index.get_level_values(0)
        p_b = prof.xs("business" if "business" in day_types else day_types[0])[cell]
        peak_slot = int(p_b.idxmax())
        base = max(float(p_b.min()), 1e-9)
        am = s[biz & (hours >= 7) & (hours < 10)].mean()
        pm = s[biz & (hours >= 16) & (hours < 19)].mean()
        wkd = s[biz].mean()
        wke = s[~biz].mean()
        rows[cell] = {
            "mean_presence": s.mean(),
            "peak_time": f"{peak_slot // 4:02d}:{(peak_slot % 4) * 15:02d}",
            "peak_to_base": float(p_b.max()) / base,
            "am_pm_ratio": am / pm if pm else np.nan,
            "weekday_weekend": wkd / wke if wke else np.nan,
            "night_share": s[hours < 6].sum() / s.sum() if s.sum() else np.nan,
            "cv": s.std() / s.mean() if s.mean() else np.nan,
        }
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "cell"
    return out


def activity_levels(wide: pd.DataFrame) -> pd.Series:
    """Classify cells as low / medium / high / very_high activity.

    A cell's 75th percentile is compared with the global quartiles of all
    observations, as in the original notebook.
    """
    allv = wide.to_numpy(dtype=float).ravel()
    q25, q50, q75 = np.nanpercentile(allv, [25, 50, 75])
    p75 = wide.quantile(0.75)
    levels = pd.cut(
        p75,
        bins=[-np.inf, q25, q50, q75, np.inf],
        labels=["low", "medium", "high", "very_high"],
    )
    return levels.rename("activity_level")


def weekly_signature(wide: pd.DataFrame) -> pd.DataFrame:
    """Average presence per (weekday, slot), each cell scaled to unit mean.

    Rows are cells and the 672 columns describe the shape of a typical week.
    """
    key = wide.index.dayofweek * SLOTS_PER_DAY + _slot(wide.index)
    sig = wide.groupby(key).mean().reindex(range(SLOTS_PER_WEEK)).interpolate().bfill().ffill()
    sig = sig / sig.mean()
    return sig.T


def area_typologies(wide: pd.DataFrame, n_types: int = 4, seed: int = 0) -> pd.DataFrame:
    """Cluster cells by the *shape* of their weekly rhythm (k-means on signatures).

    The result has the cluster id and a heuristic label derived from the cluster's
    centroid, so typologies can be related to land use (where people work, live,
    spend leisure time or change mode).
    """
    from sklearn.cluster import KMeans

    sig = weekly_signature(wide)
    k = min(n_types, len(sig))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(sig.to_numpy())
    labels = {c: _label_centroid(km.cluster_centers_[c]) for c in range(k)}
    return pd.DataFrame(
        {"typology_id": km.labels_, "typology": [labels[c] for c in km.labels_]},
        index=sig.index,
    )


def _label_centroid(centroid: np.ndarray) -> str:
    week = centroid.reshape(7, SLOTS_PER_DAY)
    weekday, weekend = week[:5].mean(axis=0), week[5:].mean(axis=0)
    hours = np.arange(SLOTS_PER_DAY) / 4
    night = weekday[hours < 6].mean()
    day = weekday[(hours >= 9) & (hours < 17)].mean()
    am = weekday[(hours >= 7) & (hours < 9.5)].mean()
    mid = weekday[(hours >= 11) & (hours < 15)].mean()
    pm = weekday[(hours >= 17) & (hours < 19.5)].mean()
    if weekend.mean() > 1.3 * weekday.mean():
        return "leisure"
    if night > day:
        return "residential"
    if am > 1.15 * mid and pm > 1.15 * mid:
        return "transport hub"
    if weekday.mean() > 1.3 * weekend.mean():
        return "workplace"
    return "mixed use"


def anomaly_scores(wide: pd.DataFrame, history_days: int | None = None) -> pd.DataFrame:
    """Robust z-scores of presence against the typical weekday/slot value.

    Large positive values flag unusual crowding, e.g. events, disruptions or
    tourist peaks (cf. Cecaj & Mamei 2017, *Data fusion for city life event detection*).
    """
    ref = wide if history_days is None else wide.iloc[: history_days * SLOTS_PER_DAY]
    key_ref = ref.index.dayofweek * SLOTS_PER_DAY + _slot(ref.index)
    med = ref.groupby(key_ref).median()
    mad = ref.groupby(key_ref).agg(lambda s: np.median(np.abs(s - np.median(s))))
    key = wide.index.dayofweek * SLOTS_PER_DAY + _slot(wide.index)
    m = med.reindex(key).to_numpy()
    d = mad.reindex(key).to_numpy() * 1.4826
    z = (wide.to_numpy(dtype=float) - m) / np.where(d > 0, d, np.nan)
    return pd.DataFrame(z, index=wide.index, columns=wide.columns)
