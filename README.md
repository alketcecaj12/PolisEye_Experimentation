## POLIcy Support systEm for smart citY data governancE

### Project PolisEye - Code description .

Every notebook already contains comments of the code that hopefully are well explained and described.

1-The data (not included) cover the area of Bologna's aeroport and that of the city of Modena, Emilia Romagna, Italy.

2- The Code

The folders as below contain code for the following tasks:

a- DataViz_ConfrontoMetodi contains notebooks that read the prediction errors and display graphics for 1Step and Multistep forecasting:

    - CompareMethods_by_ForcastError_4_1StepPrediction.ipynb
    - CompareMethods_by_ForcastError_4_MultistepPrediction.ipynb
    - ...


b- EDA contains:
   - a notebook that can ne used to fix the data and replace missing values by using backfill e forward fill methods.
     as for example the notebook FixData.ipynb
   - a notebook for exploring and visualizing data as in EDA
     ExplorativeDataAnalysis_1.ipynb
     ExplorativeDataAnalysis_Modena.ipynb
   - Timeseries Decomposition using methods from the statsmodels library
     TimeseriesDecomposition.ipynb
   - notebook for computing the mean(on a hourly and daily basis) and using it as a predictor.
     DataAggregation_4_HeatMap_and_mean_computation.ipynb  

c- DeepLearningModels notebooks conain methods for:
   - 1-step forcasting on univariate time series using neural networks as CNN, MLP e LSTM
     like for example the notebook in CNN_4_UnivariateSeries_1_Step_Forcast.ipynb
   - multistep forecasting for univariate time series by using neural networks like CNN, MLP e LSTM
     An example can be found in LSTM_4_UnivariateTimeseries_Multistep_forcast.ipynb
   - 1-step forecasting for multivariate time series with neural networks like CNN, MLP e LSTM
     The following notebook contains an example LSTM_4_MultivariateTimeseries_1_Step.ipynb
   - multistep forecasting for multivariate time series with CNN, MLP e LSTM
     CNN_4_Multivariate_multistep_time_series_forcast.ipynb

d- Prophet : contains two notebooks with code using Facebook Prophet forecasting framework.

  - Prophet - Documentation and Example.ipynb  uses example data to showcase predictions.
  - Prophet_On_Grid_Data.ipynb this notebook uses Prophet for delivering predictions real data

e-StatisticalMethods contains notebooks implementing statistical forecasting methods :

  - MultivariateSeriesPrediction
     1step forecast for multivariate time series using statistical methods like VAR
  - UnivariateSeriesPrediction
     forecasting for univariate time series using ARIMA, SARIMA, Exponetial Smothing, AR
  - NaiveMethods : naive methods like mean on daily basis or the mean on weekly basis to make predictions.

f- SpatioTemporalCA: the notebooks contained in this folder deliver spatio-temporal analysis such as:
   - Pearson correlation on cells
   - Measure causality like Granger Causality between cells
   - Time Lagged Cross Correlation
   - and Window Lagged Cross Correlation
