
# %%
from copyreg import pickle
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
import math
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
import joblib
from datetime import timedelta, datetime
import pyodbc
import sys
from tensorflow.keras.optimizers import Nadam


# %%

def connect_to_database():
    conn = pyodbc.connect('Driver={SQL Server Native Client 11.0};'
                          'Server=ZAKIR;'
                          'Database=keolis;'
                          'Trusted_Connection=yes;')
    cursor = conn.cursor()
    return cursor, conn


def input_data(line_number, direction, stop):
    cursor, conn = connect_to_database()
    # import boarding data
    occupancy_data = pd.read_sql_query(
        'select * from occupancy_per_stop where system_linenr = {} and direction = {} and stop = {}'.format(line_number, direction, stop), conn)
    occupancy_data['hour'] = pd.to_datetime(occupancy_data['date_time']).dt.hour
    # import weather data
    weather_data = pd.read_sql_query('select * from weather_data', conn)

    # school holidays
    school_holidays = pd.read_sql_query('select * from school_holidays', conn)
    school_holidays['school_holiday'] = 1

    christmas_holidays = pd.read_sql_query(
        'select * from christmas_holidays', conn)
    christmas_holidays['christmas'] = 1

    # import public holidays
    public_holidays = pd.read_sql_query('select * from public_holidays', conn)
    public_holidays['public_holiday'] = 1

    # import bus cancelation
    bus_cancellations = pd.read_sql_query(
        'select * from bus_cancellations where system_linenr = {} and direction = {}'.format(line_number, direction, stop), conn)
    bus_cancellations.drop(['system_linenr', 'direction'],axis =1,  inplace=True)
    # combine datasets
    data = pd.DataFrame(
        pd.merge(occupancy_data, weather_data, on=['date_time', 'date_time']))
    data['Date'] = data['date_time'].dt.date
    data = pd.merge(data, school_holidays, on='Date',  how='left')
    data['school_holiday'] = data['school_holiday'].fillna(0)
    data = pd.merge(data, public_holidays, on='Date',  how='left')
    data['public_holiday'] = data['public_holiday'].fillna(0)

    data = pd.merge(data, bus_cancellations, on='date_time', how='left')
    data['num_cancellations'] = data['num_cancellations'].fillna(0)

    data = data[data['date_time'].dt.hour >= 6]
    data = data[data['date_time'].dt.hour < 23]
    data['hour'] = data['date_time'].dt.hour
    column = ['date_time', 'system_linenr', 'direction', 'stop', 'occupancy',
              'temp', 'sun_duration', 'prec_duration', 'prec_amount', 'cloud_cover',
              'humidity', 'rain', 'snow', 'windx', 'windy', 'school_holiday', 'public_holiday', 'num_cancellations', 'hour']
    data = data[column]

    cursor.close()

    return data
data = input_data(4709, 1, 9920)
#%%
# def model_inputs(data):

def MLP_model_weekdays(line, direction, stop):
    data = input_data(line, direction, stop)
    data = data[data['date_time'].dt.year == 2022]
    data = data.sort_values('date_time')
    data.drop(['system_linenr', 'direction', 'stop'], axis=1, inplace=True)
    data['Date'] = data['date_time'].dt.date

    data['day'] = data['date_time'].dt.day_name()
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    # weekdays data
    weekdays = pd.DataFrame(data.loc[(data.day.isin(weekdays)), :])
    weekdays['monday'] = weekdays['day'].apply(
        lambda x: 1 if x == 'Monday' else 0)
    weekdays['tuesday'] = weekdays['day'].apply(
        lambda x: 1 if x == 'Tuesday' else 0)
    weekdays['wednesday'] = weekdays['day'].apply(
        lambda x: 1 if x == 'Wednesday' else 0)
    weekdays['thursday'] = weekdays['day'].apply(
        lambda x: 1 if x == 'Thursday' else 0)
    weekdays['friday'] = weekdays['day'].apply(
        lambda x: 1 if x == 'Friday' else 0)
    weekdays.drop(['day', 'Date'], axis=1, inplace=True)

    dataset = weekdays.set_index('date_time')
    X = np.array(dataset.drop('occupancy', axis=1))
    y = np.array(dataset['occupancy'])

    # Scalling data from 0 to 1
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_scaled = (X_scaler.fit_transform(X))
    y_scaled = (y_scaler.fit_transform(y.reshape(-1, 1)))

    # train and test split
    n = np.max(np.shape(y))
    m_train = int(0.9*n)
    m_test = n-m_train
    # merging, shuffeling and splitting the data into a training set and a test set
    data_set = np.column_stack([y_scaled, X_scaled])
    # np.random.shuffle(data_set)
    data_set_train = data_set[:m_train, :]
    data_set_test = data_set[m_train:, :]
    X_train = data_set_train[:, 1:]
    y_train = data_set_train[:, 0]
    X_test = data_set_test[:, 1:]
    y_test = data_set_test[:, 0]

    MLP_model = MLPRegressor(max_iter=10000, random_state=0)
    check_parameters = {
        'hidden_layer_sizes': [(128, 64, 32, 16), (64, 32, 16, 8), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.1, 0.01, 0.001],
        'learning_rate': ['constant', 'adaptive']
    }
    gridsearchcv = GridSearchCV(
        estimator=MLP_model, param_grid=check_parameters, n_jobs=-1, cv=3)
    
    gridsearchcv.fit(X_train, y_train)
    
    # save the model 
    name = 'MLP_{}_{}_{}.pkl'.format(line, direction, stop)
    joblib.dump(grid.best_estimator_, name, compress = 1)
    
    print('Best parameters found:\n', gridsearchcv.best_params_)

    y_true, y_pred = y_test, gridsearchcv.predict(X_test)

    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_true = y_scaler.inverse_transform(y_true.reshape(-1, 1))

    mae_weekdays = mean_absolute_error(y_true, y_pred)

    
    y_true, y_pred = y_test, grid.predict(X_test)

    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_true = y_scaler.inverse_transform(y_true.reshape(-1, 1))
    

    return y_pred, y_true


y_pred, y_true = MLP_model_weekdays(4709, 1, 9920)

predictions = np.column_stack([y_true, y_pred])
predictions = pd.DataFrame(predictions, columns=[
                           'actual_values', 'predicted_values'])
predictions['predicted_values'] = round(predictions['predicted_values'])

plt.rcParams['figure.figsize'] = (12, 7)
plt.title('Predictions vs actual values', fontsize=12)
plt.plot(predictions.actual_values, linewidth=1.5,
         marker='o', markersize=4, linestyle="--", label='Actual values')
plt.plot(predictions.predicted_values, linewidth=1.5,
         marker='o', markersize=4, linestyle="-", label='Predicted Values')


# plt.yticks(np.arange(0, 4500, step=1000))
plt.ylabel('Passengers / hour', fontsize=11)
plt.xlabel('Time [hour]', fontsize=11)
plt.legend(fontsize=10, loc=1)
plt.show()


#%%
dataset = weekdays.set_index('date_time')
date = pd.DataFrame(weekdays['date_time'])

X = np.array(dataset.drop('boarders', axis=1))
y = np.array(dataset['boarders'])

# Scalling data from 0 to 1
X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = (X_scaler.fit_transform(X))
y_scaled = (y_scaler.fit_transform(y.reshape(-1, 1)))

# train and test split
n = np.max(np.shape(y))
m_train = int(0.8*n)
m_test = n-m_train
# merging, shuffeling and splitting the data into a training set and a test set
data_set = np.column_stack([y_scaled, X_scaled])
# np.random.shuffle(data_set)
data_set_train = data_set[:m_train, :]
data_set_test = data_set[m_train:, :]
X_train = data_set_train[:, 1:]
y_train = data_set_train[:, 0]
X_test = data_set_test[:, 1:]
y_test = data_set_test[:, 0]

n_features = X_train.shape[1]


def create_model(activation='relu'):
    model = Sequential()
    model.add(Dense(256, input_dim=19,
              kernel_initializer='normal', activation=activation))
    model.add(Dense(128, kernel_initializer='normal', activation=activation))
    model.add(Dense(64, kernel_initializer='normal', activation=activation))
    model.add(Dense(1))
    # Compile model
    optimizer = Nadam(learning_rate=0.01)
    model.compile(loss='mse', optimizer=optimizer)
    return model


model = KerasRegressor(build_fn=create_model,
                       batch_size=32, epochs=50, verbose=0)

activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid']
param_grid = dict(activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid,
                    n_jobs=-1, cv=3)
grid.fit(X_train, y_train)


filename = 'test.sav'
joblib.dump(grid, filename)
# summarize results
print("Best: %f using %s" % (grid.best_score_, grid.best_params_))
means = grid.cv_results_['mean_test_score']
stds = grid.cv_results_['std_test_score']
params = grid.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

y_true, y_pred = y_test, grid.predict(X_test)

y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
y_true = y_scaler.inverse_transform(y_true.reshape(-1, 1))

mae6 = mean_absolute_error(y_true, y_pred)


# prediction vs paramter

predictions = np.column_stack([y_true, y_pred])
predictions = pd.DataFrame(predictions, columns=[
                           'actual_values', 'predicted_values'])
predictions['predicted_values'] = round(predictions['predicted_values'])
date = date.reset_index()
date.drop('index', axis=1, inplace=True)
date_test = date.loc[m_train:, :]
date_test = date_test.reset_index()
date_test.drop('index', axis=1, inplace=True)
predictions = pd.concat([predictions, date_test], axis=1)


plt.rcParams['figure.figsize'] = (12, 7)
plt.title('Predictions vs actual values', fontsize=12)
plt.plot(predictions.date_time, predictions.actual_values, linewidth=1.5,
         marker='o', markersize=4, linestyle="--", label='Actual values')
plt.plot(predictions.date_time, predictions.predicted_values, linewidth=1.5,
         marker='o', markersize=4, linestyle="-", label='MLP base model')


# plt.yticks(np.arange(0, 4500, step=1000))
plt.ylabel('Passengers / hour', fontsize=11)
plt.xlabel('Time [hour]', fontsize=11)
plt.legend(fontsize=10, loc=1)
plt.show()

a = predictions['predicted_values'].sum()
b = predictions['actual_values'].sum()

dataset = weekdays.set_index('date_time')
date = pd.DataFrame(weekdays['date_time'])

X = np.array(dataset.drop('boarders', axis=1))
y = np.array(dataset['boarders'])


# Scalling data from 0 to 1
X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = (X_scaler.fit_transform(X))
y_scaled = (y_scaler.fit_transform(y.reshape(-1, 1)))


# for column in ['temp', 'sun_duration', 'prec_duration', 'prec_amount', 'humidity']

# train and test split
n = np.max(np.shape(y))
m_train = int(0.9*n)
m_test = n-m_train
# merging, shuffeling and splitting the data into a training set and a test set
data_set = np.column_stack([y_scaled, X_scaled])
# np.random.shuffle(data_set)
data_set_train = data_set[:m_train, :]
data_set_test = data_set[m_train:, :]
X_train = data_set_train[:, 1:]
y_train = data_set_train[:, 0]
X_test = data_set_test[:, 1:]
y_test = data_set_test[:, 0]

# number of features
n_features = X_train.shape[1]

MLP_model_default = MLPRegressor(max_iter=10000, random_state=0)
check_parameters = {
    'hidden_layer_sizes': [(128, 64, 32, 16), (64, 32, 16, 8), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.1, 0.01, 0.001],
    'learning_rate': ['constant', 'adaptive']
}
gridsearchcv = GridSearchCV(
    estimator=MLP_model_default, param_grid=check_parameters, n_jobs=-1, cv=3)
gridsearchcv.fit(X_train, y_train)


print('Best parameters found:\n', gridsearchcv.best_params_)

y_true, y_pred_c2 = y_test, gridsearchcv.predict(X_test)

y_pred_c2 = y_scaler.inverse_transform(y_pred_c2.reshape(-1, 1))
y_true = y_scaler.inverse_transform(y_true.reshape(-1, 1))

mae_weekdays_c2 = mean_absolute_error(y_true, y_pred_c2)


predictions2 = np.column_stack([y_true, y_pred_c2])
predictions2 = pd.DataFrame(predictions2, columns=[
                            'actual_values', 'predicted_values'])
date = date.reset_index()
date.drop('index', axis=1, inplace=True)
date_test = date.loc[m_train:, :]
date_test = date_test.reset_index()
date_test.drop('index', axis=1, inplace=True)
predictions2 = pd.concat([predictions2, date_test], axis=1)
predictions2['predicted_values'] = round(predictions2['predicted_values'])


plt.rcParams['figure.figsize'] = (12, 7)
plt.title('Predictions vs actual values', fontsize=12)
sns.lineplot(predictions2.date_time, predictions2.actual_values, linewidth=1.5,
             marker='o', markersize=4, linestyle="--", label='Actual values')
sns.lineplot(predictions2.date_time, predictions2.predicted_values, linewidth=1.5,
             marker='o', markersize=4, linestyle="-", label='MLP base model')

# plt.yticks(np.arange(0, 4500, step=1000))
plt.ylabel('Passengers / hour', fontsize=11)
plt.xlabel('Time [hour]', fontsize=11)
plt.legend(fontsize=10, loc=1)
plt.show()

# test = boarding_data[boarding_data['date_time'].dt.year == 2019]

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
# ax1.plot(test.date_time, test.boarders, color='green')
# ax1.legend(['direction 1'], loc=2)
# # ax2.plot(test2.OperatingDate,
# #          test2.BoardersCorrected, color='blue')
# # ax2.legend(['Planned departure'], loc=2)
# # ax2.set(xlabel='Departure Time', ylabel='Occupancy')
# # ax2.title.set_text('2022-02-11 \n trip number: {}'.format(trip_number))
# fig.tight_layout()
# plt.show()
