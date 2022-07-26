import sys 
sys.path.insert(0, 'M:/optimization/bus_reassignment_problem/bus_reassignment') #add path to the file

import data_preprocessing as dp

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import math
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
import pyodbc 
import joblib
from datetime import timedelta, datetime



def import_data():
    cursor, conn = dp.connect_to_database()
    bus_lines = pd.read_sql_query('select * from bus_lines', conn)

    boarding_data = pd.read_sql_query('select * from trip_data2019', conn)

    boarding_data['DateTime'] = boarding_data.apply(
        lambda row: row['DateTime'] + timedelta(hours=1), 
        axis=1)

    trip_2018 = pd.read_sql_query('select * from trip_data2018', conn)
    trip_2018['DateTime'] = trip_2018.apply(
        lambda row: row['DateTime'] + timedelta(hours=1), 
        axis=1)

    christmas_holidays = pd.read_sql_query('select * from christmas_holidays', conn)
    christmas_holidays['christmas']=1

    school_holidays = pd.read_sql_query('select * from school_holidays', conn)
    school_holidays['school_holiday'] = 1

    public_holidays = pd.read_sql_query('select * from public_holidays', conn)
    public_holidays['public_holiday'] = 1

    bus_cancellation = pd.read_sql_query('select * from bus_cancellation', conn)
    bus_cancellation = pd.DataFrame(bus_cancellation.groupby(['DateTime'], as_index=False)['bus_cancellation'].sum())

    #football = pd.concat((football_2018, football_2019))
    football = pd.read_sql_query('select * from football', conn)

    weather = pd.read_sql_query('select * from weather_data', conn)
    weather['prec_amount'] = weather['prec_amount'].apply(lambda x: 0 if x < 0 else x)
    weather['DateTime'] =  pd.to_datetime(weather['DateTime'], infer_datetime_format=True)
