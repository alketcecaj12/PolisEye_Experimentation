### Descrizione del codice contenuto nelle varie cartelle/notebook .

Ogni notebook contiene commenti e descrizione del codice che spero siano abbastanza chiari e completti.

1-I DATI

Dati (aeroporto e Modena) da usare con il codice si trovano in questa cartella su Drive :
: https://drive.google.com/drive/folders/1g3wISpUgCMOpDIFLXeqoCSVaw1Nw8hNG?usp=sharing


2- IL CODICE

Il progetto su github(la repo è privata) si trova in questo link: https://github.com/alketcecaj12/poliseye

La cartella (in seguito enummerate come in a, b, c):

a- DataViz_ConfrontoMetodi contiene due notebook che fanno il confronto tra i metodi in base all'errore
   di previsione 1Step e Multistep:

    - CompareMethods_by_ForcastError_4_1StepPrediction.ipynb
    - CompareMethods_by_ForcastError_4_MultistepPrediction.ipynb


b- EDA contiene
   - notebook per sistemare i dati (valori anomali sostituiti con backfill e forward fill)
     ad esempio il notebook FixData.ipynb
   - notebook per visualizzare i dati in filisofia EDA
     ExplorativeDataAnalysis_1.ipynb
     ExplorativeDataAnalysis_Modena.ipynb
   - scomposizione di serie temporali usando metodi di scomposizione
     TimeseriesDecomposition.ipynb
   - notebook per il calcolo della media (in base oraria e giornaliera) per poi usarla come predictor.
     DataAggregation_4_HeatMap_and_mean_computation.ipynb  

c- DeepLearningModels contiene notebook per:
   - previsone 1-step per serie univariate con reti neurali CNN, MLP e LSTM
     come ad esempio il CNN_4_UnivariateSeries_1_Step_Forcast.ipynb
   - previsone multistep per serie univariate con reti neurali CNN, MLP e LSTM
     LSTM_4_UnivariateTimeseries_Multistep_forcast.ipynb
   - previsone 1-step per serie multivariate con reti neurali CNN, MLP e LSTM
     LSTM_4_MultivariateTimeseries_1_Step.ipynb
   - previsone multistep per serie multivariate con reti neurali CNN, MLP e LSTM
     CNN_4_Multivariate_multistep_time_series_forcast.ipynb

d- img : contiene imagini e grafici

e- Prophet : contiene due notebook  e il secondo
   usa i dati per fare previsone.
  - Prophet - Documentation and Example.ipynb  si riferisce alla documentazione (esempio)
  - Prophet_On_Grid_Data.ipynb che usa Prophet per fare previsione sui dati

f-StatisticalMethods che contiene notebook per :

  - MultivariateSeriesPrediction
     previsione 1step usando metodi statistici come VAR
  - UnivariateSeriesPrediction
     previsone usando ARIMA, SARIMA, Exponetial Smothing, AR
  - NaiveMethods :
    Contiene metodi naive come autoregression e media stagionale

g- SpatioTemporalCA: contiene notebook con metodi per l'analis spaziotemporale dei dati
   - correlazione usando Pearson
   - come la GrangerCausality
   - Time Lagged Cross Correlation
   - oppure Window Lagged Cross Correlation
