import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import datetime
from random import random
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.ar_model import AR
import warnings
from statsmodels.tsa.arima_model import ARIMA
warnings.filterwarnings('ignore')



# grid search ARIMA parameters for time series
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()

    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse


# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values): 
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order 
                    #print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                      continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))
    l = []
    l = best_cfg, best_score
    return l

if __name__ == '__main__':

    data = pd.read_csv('/Users/alket/Desktop/dati/new_data_backfill_forwfill.csv',index_col = 0)
    gbc = data.groupby(by = data['cell_num'])
    data2dict = {}
    count = 0
    for index, k_df in gbc:

        count +=1
        cell_number = index
        print(count, cell_number)
        if count > 3: break
        # model configs

        series = k_df['nr_people'][0:672]
        print(len(series))
        p_values = [0, 2, 4, 8]
        d_values = range(0, 3)
        q_values = range(0, 3)
        warnings.filterwarnings("ignore")
        ls = evaluate_models(series.values, p_values, d_values, q_values)
        data2dict[cell_number]=ls

with open('BestARIMA_config_parametres.csv', 'w') as f:
    for key, value in data2dict.items():
        f.write('%s:%s\n' % (key, value))