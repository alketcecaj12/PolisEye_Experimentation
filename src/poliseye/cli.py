"""Command line entry point: ``poliseye``.

Examples
--------
Run the benchmark and travel-behaviour profiling on synthetic data::

    poliseye demo

Run them on your own presence file (processed or raw layout)::

    poliseye benchmark data.csv --horizon 16 --cells 486-1258 488-1264
    poliseye profile data.csv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from poliseye.backtest import compare_models, leaderboard
from poliseye.baselines import (
    DayHourProfileForecaster,
    MovingAverageForecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)
from poliseye.data import SLOTS_PER_DAY, clean_presence, load_presence, synthetic_presence, to_wide
from poliseye.mobility import activity_levels, area_typologies, mobility_indicators


def default_models(include_statistical: bool = False) -> dict:
    models = {
        "persistence": PersistenceForecaster,
        "seasonal naive (1 day)": SeasonalNaiveForecaster,
        "same slot, last 4 days": lambda: MovingAverageForecaster(4, SLOTS_PER_DAY),
        "weekday x slot profile": DayHourProfileForecaster,
    }
    if include_statistical:
        from poliseye.statistical import ETSForecaster

        models["Holt-Winters (daily)"] = ETSForecaster
    return models


def _load(path: str | None) -> pd.DataFrame:
    if path is None:
        return synthetic_presence()
    return clean_presence(to_wide(load_presence(path)))


def cmd_benchmark(args) -> None:
    wide = _load(args.data)
    res = compare_models(
        wide, default_models(args.statistical), args.horizon, args.origins, cells=args.cells
    )
    print(f"Rolling-origin benchmark, horizon {args.horizon} x 15 min, {args.origins} origins\n")
    print(leaderboard(res).round(2).to_string())


def cmd_profile(args) -> None:
    wide = _load(args.data)
    table = (
        mobility_indicators(wide)
        .join(activity_levels(wide))
        .join(area_typologies(wide, n_types=args.types))
    )
    print(table.drop(columns="typology_id").round(2).to_string())


def cmd_demo(args) -> None:
    print("Synthetic grid: four cells with archetypal daily rhythms\n")
    cmd_profile(argparse.Namespace(data=None, types=4))
    print()
    cmd_benchmark(
        argparse.Namespace(data=None, horizon=4, origins=7, cells=None, statistical=False)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="poliseye", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("benchmark", help="compare forecasters with rolling-origin evaluation")
    b.add_argument("data", nargs="?", help="presence CSV (synthetic data if omitted)")
    b.add_argument("--horizon", type=int, default=4)
    b.add_argument("--origins", type=int, default=7)
    b.add_argument("--cells", nargs="*")
    b.add_argument("--statistical", action="store_true", help="also fit Holt-Winters")
    b.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("profile", help="travel-behaviour indicators and area typologies")
    p.add_argument("data", nargs="?", help="presence CSV (synthetic data if omitted)")
    p.add_argument("--types", type=int, default=4)
    p.set_defaults(func=cmd_profile)

    d = sub.add_parser("demo", help="benchmark + profile on synthetic data")
    d.set_defaults(func=cmd_demo)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
