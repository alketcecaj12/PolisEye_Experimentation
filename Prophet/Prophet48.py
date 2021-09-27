import datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from fbprophet import Prophet
import math
# additive decompose a contrived additive time series

from statsmodels.tsa.seasonal import seasonal_decompose 

# the main library has a small set of functionality
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive,
                                         drift, 
                                         mean, 
                                         seasonal_naive)

# funzioni per preparare i dati e calcolare le previsioni
def prepare_data_4_STL(serie_dati):
    counter = 0
    dict2data = {}
    error_list = []
    print(counter)
    counter +=1
    dates4dec = []
    cell_values = []

    for index, row in serie_dati.iterrows():
        date = row['date']
        h = str(row['hours'])
        h = h.split('.')
    
        if len(h[0])<2:
            h = h[1]+h[0]
        else: 
            h = h[0]
   
        minutes = str(row['minutes'])
        m = ''
        minutes = minutes.split('.')
        if len(minutes[0])<2: 
            m = minutes[0] +'0'
        else: 
            m = minutes[0]
        data_f = date+' '+h+':'+m+':'+'00'
        cell_values.append(row['nr_people'])
        dates4dec.append(data_f)
    dict_i = {'ds': dates4dec, 'y':cell_values}
    data4deco = pd.DataFrame(dict_i, index=None, columns=None)  
    data4deco['ds'] = pd.to_datetime(data4deco['ds'])
    data4deco = data4deco.set_index('ds')
    return data4deco    
    

def get_forcast_per_component(data4prophet, train_test_size):
    
    # fit the model
    m = Prophet(changepoint_prior_scale=0.01).fit(data4prophet)
    
    # make predictions
    future = m.make_future_dataframe(periods=0, freq='H')
    
    # make forcast.. 
    fcast = m.predict(future)
    
    fcast = fcast[train_test_size:]
    return fcast
        
    
# carica dati
data = pd.read_csv('/Users/alket/Desktop/dati/new_data_backfill_forwfill.csv', index_col = 0, 
                   header=0, parse_dates=True)

# aggrega dati
agg_by_cell = data.groupby(by = ['cell_num'])

# dichiara counter e struttura dati per i dati d'errore per cella
counter = 0
dict2data = {}
dict2MAPE = {}
dict2RMSE = {}
counter_i = 0

# ittera per tutte le celle
for ii, kk in agg_by_cell:
    print(counter_i)
    counter_i+=1
    #if counter_i > 6: break
    kk = kk.iloc[::4, :]  
    cell_num = ii    
    data4decom = prepare_data_4_STL(kk)
    decomped_data = decompose(data4decom['y'], period=96)

    #print(decomped_data.trend)
    data_trend = decomped_data.trend.to_frame()
    data_seasonal = decomped_data.seasonal.to_frame()
    data_residual = decomped_data.resid.to_frame()
    
    data_trend = data_trend.reset_index()
    data_seasonal = data_seasonal.reset_index()
    data_residual = data_residual.reset_index()
    
    train_test_size = 2600
    steps = 48
    steps2 = 96
    steps3 = 144
    steps4 = 192
    steps5 = 240
    #train_test_size += steps 
    forcasted_trend = get_forcast_per_component(data_trend, train_test_size)
    forcasted_residual = get_forcast_per_component(data_residual, train_test_size)
    forcasted_season = get_forcast_per_component(data_seasonal, train_test_size)
    
    forcasted_trend = np.array(forcasted_trend['yhat'].tolist())
    forcasted_season = np.array(forcasted_season['yhat'].tolist())
    forcasted_residual = np.array(forcasted_residual['yhat'].tolist())
    
    # combina le previsioni 
    final_prediction = forcasted_trend + forcasted_residual + forcasted_season
    #print(final_prediction)
    X, y = kk[:train_test_size], kk[train_test_size:]
    
    # assegna a expected il valore del test set
    expected = y['nr_people'].values
    
    expected1 = expected[:steps]
    final_prediction1 = final_prediction[:steps]
    
    expected2 = expected[steps:steps2]
    final_prediction2 = final_prediction[steps:steps2]
    
    expected3 = expected[steps2:steps3]
    final_prediction3 = final_prediction[steps2:steps3]
    
    expected4 = expected[steps3:steps4]
    final_prediction4 = final_prediction[steps3:steps4]
    
    expected5 = expected[steps4:steps5]
    final_prediction5 = final_prediction[steps4:steps5]
    
    f_expected = expected1 +expected2+ expected3+expected4+expected5
    f_predicted = final_prediction1 + final_prediction2+ final_prediction3+final_prediction4+final_prediction5
    
    print('exp', len(expected))
    print('fp', len(final_prediction))
    # calcola differenza (errore) tra predicted e expected 
    difference = abs(f_expected - f_predicted)
    
    pow_err = np.power(difference, 2)
    rmse = math.sqrt(np.mean(pow_err))
    dict2RMSE[cell_num]=rmse
    
    #sMAPE = np.mean(200*(difference/(expected - final_prediction)))
    
    MAPE = np.mean(abs(100*(difference/f_expected)))
    dict2MAPE[cell_num] = MAPE
    # calcola errore medio e altre misure 
    mean_error = difference # np.reshape(difference, difference.shape[0] * difference.shape[1])
    
    # collect data 2 dictionary
    minimum = np.amin(mean_error)   
    per75 = np.percentile(mean_error, 75)
    per50 = np.percentile(mean_error, 50)
    per25 = np.percentile(mean_error, 25)
    maximum = np.amax(mean_error)
    l5i = [minimum, per25, per50, per75, maximum]
    
    dict2data[cell_num] = l5i
    
    
with open('MAE_error_data_4_Prophet_with_STL_Decomposition_48.csv', 'w') as f:
    for key, value in dict2data.items():
        f.write('%s:%s\n' % (key, value)) 
        
with open('MAPE_error_data_4_Prophet_with_STL_Decomposition_48.csv', 'w') as f:
    for key, value in dict2MAPE.items():
        f.write('%s:%s\n' % (key, value))   
        
with open('RMSE_error_data_4_Prophet_with_STL_Decomposition_48.csv', 'w') as f:
    for key, value in dict2RMSE.items():
        f.write('%s:%s\n' % (key, value))         