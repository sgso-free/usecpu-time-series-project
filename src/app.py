import numpy as np
import pandas as pd
import pickle

from pmdarima.arima import auto_arima  

##*********** MODEL A **************
#Load Data
train_dataA = pd.read_csv('/workspace/usecpu-time-series-project/data/raw/cpu-train-a.csv')
test_dataA = pd.read_csv('/workspace/usecpu-time-series-project/data/raw/cpu-test-a.csv')

#Convert the dataframe index to a datetime index 
df_raw_A = train_dataA.copy()
df_raw_A['date'] = pd.to_datetime(df_raw_A['datetime'])
df_raw_A = df_raw_A.set_index('date')
df_raw_A.drop(['datetime'], axis=1, inplace=True)

#Convert the dataframe index to a datetime index 
df_test_A = test_dataA.copy()
df_test_A['date'] = pd.to_datetime(df_test_A['datetime'])
df_test_A = df_test_A.set_index('date')
df_test_A.drop(['datetime'], axis=1, inplace=True) 

stepwise_model_A = auto_arima(df_raw_A, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

# Train the model
stepwise_model_A.fit(df_raw_A)
  
# Save the model as a pickle 
filename = '/workspace/usecpu-time-series-project/models/model_a.pkl'
pickle.dump(stepwise_model_A, open(filename,'wb'))


##*********** MODEL B **************

#Load Data
train_dataB = pd.read_csv('/workspace/usecpu-time-series-project/data/raw/cpu-train-b.csv')
test_dataB = pd.read_csv('/workspace/usecpu-time-series-project/data/raw/cpu-test-b.csv')

#Convert the dataframe index to a datetime index 
df_raw_B = train_dataB.copy()
df_raw_B['date'] = pd.to_datetime(df_raw_B['datetime'])
df_raw_B = df_raw_B.set_index('date')
df_raw_B.drop(['datetime'], axis=1, inplace=True)

#Convert the dataframe index to a datetime index 
df_test_B = test_dataB.copy()
df_test_B['date'] = pd.to_datetime(df_test_B['datetime'])
df_test_B = df_test_B.set_index('date')
df_test_B.drop(['datetime'], axis=1, inplace=True) 

stepwise_model_B = auto_arima(df_raw_B, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

# Train the model
stepwise_model_B.fit(df_raw_B)
  
# Save the model as a pickle 
filename = '/workspace/usecpu-time-series-project/models/model_b.pkl'
pickle.dump(stepwise_model_B, open(filename,'wb'))


###############

#load the model from data A
filenameA = '/workspace/usecpu-time-series-project/models/model_a.pkl'
load_model_a = pickle.load(open(filenameA, 'rb'))
 
predicciones_A=load_model_a.fit_predict(df_test_A,n_periods=60*24)

print('Predicciones para el próximo día Modelo A son: {}'.format(predicciones_A))

#load the model from data B
filenameB = '/workspace/usecpu-time-series-project/models/model_b.pkl'
load_model_b = pickle.load(open(filenameB, 'rb'))
 
# reentreno el modelo con la ventana móvil (historial)
predicciones_B=load_model_b.fit_predict(df_test_B,n_periods=60*24)

print('Predicciones para el próximo día Modelo B son: {}'.format(predicciones_B))