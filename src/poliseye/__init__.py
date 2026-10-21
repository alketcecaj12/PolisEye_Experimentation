"""PolisEye: human presence and travel behaviour from aggregated mobile phone data.

The package turns the research notebooks of the POLIS-EYE project into reusable,
tested building blocks:

* :mod:`poliseye.data`        loading, cleaning and reshaping cell-level presence data
* :mod:`poliseye.timefeatures` calendar features (business days, public holidays)
* :mod:`poliseye.windows`     supervised windows for machine learning forecasters
* :mod:`poliseye.metrics`     forecast error measures (MAE, RMSE, MAPE, MaxAE)
* :mod:`poliseye.baselines`   naive and seasonal-profile forecasters
* :mod:`poliseye.statistical` ARIMA, SARIMA and exponential smoothing forecasters
* :mod:`poliseye.decomposition` STL decomposition and decomposition-based forecasting
* :mod:`poliseye.backtest`    rolling-origin evaluation and model comparison
* :mod:`poliseye.spatial`     spatio-temporal dependence between cells
* :mod:`poliseye.mobility`    travel-behaviour indicators and area typologies
* :mod:`poliseye.deep`        optional MLP / CNN / LSTM forecasters (needs ``[deep]``)
"""

from poliseye.backtest import compare_models, rolling_origin
from poliseye.baselines import (
    DayHourProfileForecaster,
    MovingAverageForecaster,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
)
from poliseye.data import (
    clean_presence,
    load_cell_errors,
    load_presence,
    synthetic_presence,
    to_wide,
)
from poliseye.metrics import evaluate, mae, mape, max_ae, rmse
from poliseye.mobility import activity_levels, area_typologies, mobility_indicators

__version__ = "0.2.0"

__all__ = [
    "DayHourProfileForecaster",
    "MovingAverageForecaster",
    "PersistenceForecaster",
    "SeasonalNaiveForecaster",
    "activity_levels",
    "area_typologies",
    "clean_presence",
    "compare_models",
    "evaluate",
    "load_cell_errors",
    "load_presence",
    "mae",
    "mape",
    "max_ae",
    "mobility_indicators",
    "rmse",
    "rolling_origin",
    "synthetic_presence",
    "to_wide",
]
