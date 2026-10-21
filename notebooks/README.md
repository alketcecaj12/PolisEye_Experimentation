# Research notebooks

These are the original POLIS-EYE experiment notebooks, kept as they were written so
that every published result can be traced back to its code. Reusable logic from them
now lives in the [`poliseye`](../src/poliseye) package; new work should import from there.

> **Running them.** The data are not included (see the main README). Several notebooks
> read from absolute paths such as `/Users/alket/Desktop/dati/...`; point those at your
> own copy of the data. Some notebooks use older APIs (`fbprophet`,
> `statsmodels.tsa.arima_model`, standalone `keras`) and need the library versions of
> 2019–2020. Comments and some file names are in Italian (*confronto* = comparison,
> *previsione* = forecast, *cella* = cell, *errore* = error).

| Folder | What it contains | Package equivalent |
|---|---|---|
| [`EDA/`](EDA) | Data repair (sentinel values, forward/back fill: `FixData.ipynb`), exploration of Bologna airport and Modena (`ExplorativeDataAnalysis_*.ipynb`), seasonal decomposition (`TimeseriesDecomposition.ipynb`), hourly/daily means and heat maps (`DataAggregation_4_HeatMap_and_mean_computation.ipynb`), time-series classification. The `index_*.html` files are interactive Folium maps of presence by hour. | `poliseye.data`, `poliseye.timefeatures`, `poliseye.decomposition`, `poliseye.mobility` |
| [`StatisticalModels/NaiveMethods/`](StatisticalModels/NaiveMethods) | Naive forecasts (previous value, same weekday and hour average) and the classification of cells by activity level and profile. | `poliseye.baselines`, `poliseye.mobility.activity_levels` |
| [`StatisticalModels/UnivariateSeriesPrediction/`](StatisticalModels/UnivariateSeriesPrediction) | AR, ARIMA, SARIMA and exponential smoothing, including hyper-parameter searches and multi-step forecasts. | `poliseye.statistical` |
| [`StatisticalModels/MultivariateSeriesPrediction/`](StatisticalModels/MultivariateSeriesPrediction) | Vector autoregression across cells. | |
| [`DeepLearningModels/`](DeepLearningModels) | MLP, CNN, LSTM, CNN-LSTM and encoder-decoder forecasters for univariate and multivariate series, 1-step and multi-step, with and without STL decomposition, holiday and lag features, and the effect of the input window length. | `poliseye.deep`, `poliseye.windows` |
| [`Prophet/`](Prophet) | Prophet on example and grid data, with STL decomposition at several horizons. | `poliseye.decomposition.STLForecaster` |
| [`Confronto/`](Confronto) | Head-to-head comparisons of all methods on the same cells (univariate and `Multivariate/`). | `poliseye.backtest` |
| [`DataViz_ConfrontoMetodi/`](DataViz_ConfrontoMetodi) | Plots of forecast errors (MAE, MAPE) by method, horizon and input window. | `poliseye.backtest.leaderboard` |
| [`SpatioTemporalCA/`](SpatioTemporalCA) | Pearson correlation between cells, time-lagged and windowed cross-correlation, Granger causality. | `poliseye.spatial` |
| [`legacy_scripts/`](legacy_scripts) | Original grid-search scripts for ARIMA, SARIMA and ETS and their best configurations. | `poliseye.statistical.grid_search` |

The CSV files next to the notebooks are per-cell forecast errors (`cell:value` per
line). Load any of them with `poliseye.data.load_cell_errors`.
