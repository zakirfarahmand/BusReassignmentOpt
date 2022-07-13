import statistics
from datetime import datetime
from random import triangular
from traceback import format_exception
from unittest.util import sorted_list_difference
from click import secho
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gurobipy import tuplelist
import googlemaps
from soupsieve import select
gmaps = googlemaps.Client(key='AIzaSyAra0o3L3rs-uHn4EpaXx1Y57SIF_02684')


def import_data():
    # call data from the database
    data = pd.read_csv(
        r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/Occupancy2022/Occupancy2022.csv', sep=';')

    lines_connected_2enschede = [4701, 4702, 4703, 4704,
                                 4705, 4706, 4707, 4708, 4709, 4060, 4061, 4062]
    data = data[data['Systeemlijnnr'].isin(
        lines_connected_2enschede)]  # only relevant lines
    # fix data format
    data['ActualArrivalTime'] = pd.to_datetime(
        data['ActualArrivalTime'], format='%Y-%m-%d %H:%M:%S')
    data['ActualDepartureTime'] = pd.to_datetime(
        data['ActualDepartureTime'], format='%Y-%m-%d %H:%M:%S')
    data['ArrivalTime'] = pd.to_datetime(
        data['ArrivalTime'], format='%Y-%m-%d %H:%M:%S')
    data['DepartureTime'] = pd.to_datetime(
        data['DepartureTime'], format='%Y-%m-%d %H:%M:%S')

    # fill nan values
    data['OccupancyCorrected'] = data['Occupancy'].fillna(0)
    data.sort_values(by=['TripNumber', 'DepartureTime',
                         'Systeemlijnnr'], inplace=True)

    # for the sake of this project, only a fraction of columns are required
    column = ['TripNumber', 'Direction', 'Systeemlijnnr', 'IdVehicle', 'OperatingDate',
              'ActualDepartureTime', 'DepartureTime', 'IdDimHalte', 'Breedtegraad', 'Lengtegraad', 'OccupancyCorrected']
    data = data[column]
    return data


def select_data(year, month, day):
    data = import_data()
    data['month'] = pd.to_datetime(data['OperatingDate']).dt.month
    data['day'] = pd.to_datetime(data['OperatingDate']).dt.day
    data['year'] = pd.to_datetime(data['OperatingDate']).dt.year
    # select one day for testing
    data = data[(data['month'] == month) & (
        data['day'] == day) & (data['year'] == year)]

    return data
# calculating deadhead time


def calculate_distance(lat1, lon1, lat2, lon2):
    distance = gmaps.distance_matrix([str(lat1) + " " + str(lon1)],
                                     [str(lat2) + " " + str(lon2)],
                                     departure_time=datetime.now().timestamp(),
                                     mode='driving')['rows'][0]['elements'][0]['duration']['text'].split(' ')[0]
    distance = int(distance) * 60000
    return distance


def data_preprocessing(data):
    first_stop_dict = {}
    last_stop_dict = {}
    bus_trips_dict = {}
    line_trips_dict = {}
    dep_time_dict = {}
    arr_time_dict = {}
    travel_time_dict = {}
    first_stop = data.sort_values(by=['Systeemlijnnr', 'DepartureTime']).drop_duplicates(
        subset=['TripNumber'], keep='first')
    line_trips_dict = {k: list(v) for k, v in first_stop.groupby(
        'Systeemlijnnr')['TripNumber']}
    bus_trips_dict = {k: list(v)
                      for k, v in first_stop.groupby('IdVehicle')['TripNumber']}
    dep_time_dict = {k: v for k, v in zip(
        first_stop['TripNumber'], first_stop['DepartureTime'])}

    first_stop = data.groupby(['IdDimHalte']).nth(0).reset_index()

    first_stop_dict = {k: v for k, v in zip(
        first_stop.TripNumber, first_stop.IdDimHalte)}
    first_lat_lon = {k: (i, j) for k, i, j in zip(
        first_stop['IdDimHalte'], first_stop['Breedtegraad'], first_stop['Lengtegraad'])}

    last_stop = data.sort_values(by=['Systeemlijnnr', 'DepartureTime']).drop_duplicates(
        subset=['TripNumber'], keep='last')
    arr_time_dict = {k: v for k, v in zip(
        last_stop['TripNumber'], last_stop['DepartureTime'])}
    last_stop = last_stop.groupby(['IdDimHalte']).nth(0).reset_index()
    last_stop_dict = {k: v for k, v in zip(
        last_stop.TripNumber, last_stop.IdDimHalte)}
    last_stop = last_stop[['TripNumber',
                           'IdDimHalte', 'Breedtegraad', 'Lengtegraad']]
    last_lat_lon = {k: (i, j) for k, i, j in zip(
        last_stop['IdDimHalte'], last_stop['Breedtegraad'], last_stop['Lengtegraad'])}

    deadhead = tuplelist()
    for key1, val1 in last_lat_lon.items():
        for key2, val2 in first_lat_lon.items():
            lat1 = val1[0]
            lon1 = val1[1]
            lat2 = val2[0]
            lon2 = val2[1]
            deadhead += [((key1, key2),
                          calculate_distance(lat1, lon1, lat2, lon2))]
    deadhead_dict = dict(deadhead)

    for k1, v1 in dep_time_dict.items():
        for k2, v2 in arr_time_dict.items():
            if k1 == k2:
                travel_time = (v2 - v1).dt.total_seconds() * 1000
                travel_time_dict.update({k1: travel_time})
    # return datasets
    return deadhead_dict, line_trips_dict, bus_trips_dict, first_stop_dict, last_stop_dict, dep_time_dict, arr_time_dict, travel_time_dict


deadhead_dict, line_trips_dict, bus_trips_dict, first_stop_dict, last_stop_dict, dep_time_dict, arr_time_dict = data_preprocessing(
    select_data(2022, 2, 11))


def cal_waiting_time(mean, var):
    waiting_time = mean * 0.5 + 0.5 * (var / mean)
    return waiting_time


def waiting_time(data):
    sorted_data = data.sort_values(by=['Systeemlijnnr', 'DepartureTime']).drop_duplicates(
        subset=['TripNumber'], keep='first')

    sorted_data['hour'] = sorted_data['DepartureTime'].dt.hour
    # calculate headway for different time of the day: peak and off-peak
    sorted_data['h_headway'] = sorted_data.groupby(by=['Systeemlijnnr', 'Direction'])[
        'DepartureTime'].transform(pd.Series.diff)
    sorted_data['h_headway'] = (sorted_data['h_headway'].dt.total_seconds())/60

    # remove the first trip of each line
    sorted_data.dropna(inplace=True)
    sorted_data = sorted_data.reset_index()
    # calculate headway variance for different time of the day: peak and non-peak  hours
    data_morning_nonpeak = sorted_data[sorted_data['hour'] <= 6]
    data_morning_peak = sorted_data[sorted_data['hour'] >= 7]  # before 6:00 Am
    data_morning_peak = data_morning_peak[data_morning_peak['hour'] <= 9]
    data_mid_nonpeak = sorted_data[sorted_data['hour'] >= 10]
    data_mid_nonpeak = data_mid_nonpeak[data_mid_nonpeak['hour'] <= 16]
    data_afternoon_peak = sorted_data[sorted_data['hour'] >= 17]
    data_afternoon_peak = data_afternoon_peak[data_afternoon_peak['hour'] <= 19]
    data_evening_nonpeak = sorted_data[sorted_data['hour'] >= 20]
    data_evening_nonpeak = data_evening_nonpeak[data_evening_nonpeak['hour'] < 23]

    data_nonpeak = pd.concat(
        [data_morning_nonpeak, data_mid_nonpeak, data_evening_nonpeak], axis=0)
    data_peak = pd.concat([data_morning_peak, data_afternoon_peak])

    # drop lines with only one trip during peak or non peak
    data_nonpeak = data_nonpeak.reset_index()
    data_nonpeak_onetrip = data_nonpeak.groupby(
        ['Systeemlijnnr', 'Direction']).size().reset_index(name='count')
    data_nonpeak_onetrip = data_nonpeak_onetrip[data_nonpeak_onetrip['count'] == 1]['Systeemlijnnr'].tolist(
    )
    data_nonpeak = data_nonpeak[~data_nonpeak['Systeemlijnnr'].isin(
        data_nonpeak_onetrip)]

    data_peak = data_peak.reset_index()
    data_peak_onetrip = data_peak.groupby(
        ['Systeemlijnnr', 'Direction']).size().reset_index(name='count')
    data_peak_onetrip = data_peak_onetrip[data_peak_onetrip['count']
                                          == 1]['Systeemlijnnr'].tolist()
    data_peak = data_peak[~data_peak['Systeemlijnnr'].isin(data_peak_onetrip)]
    # calculate headway mean
    data_nonpeak['h_var'] = data_nonpeak.groupby(
        ['Systeemlijnnr', 'Direction'])['h_headway'].transform(statistics.variance)
    data_peak['h_var'] = data_peak.groupby(
        ['Systeemlijnnr', 'Direction'])['h_headway'].transform(statistics.variance)
    data_nonpeak['h_mean'] = data_nonpeak.groupby(
        ['Systeemlijnnr', 'Direction'])['h_headway'].transform(statistics.mean)
    data_peak['h_mean'] = data_peak.groupby(
        ['Systeemlijnnr', 'Direction'])['h_headway'].transform(statistics.mean)

    # calculate waiting time
    data_nonpeak['waiting_time'] = cal_waiting_time(
        data_nonpeak['h_mean'], data_nonpeak['h_var'])
    data_peak['waiting_time'] = cal_waiting_time(
        data_peak['h_mean'], data_peak['h_var'])

    sorted_data = pd.concat([data_peak, data_nonpeak], axis=0)
    waiting_time_dict = sorted_data.set_index(
        ['TripNumber']).to_dict()['waiting_time']

    return waiting_time_dict


# %%
"""
plot occupancy data
"""
data = select_data(2022, 2, 11)
trip_number = 40893
test_data = data[data['TripNumber'] == trip_number]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
# fig.tight_layout()
ax1.plot(test_data.DepartureTime, test_data.OccupancyCorrected, color='green')
ax1.legend(['Actual departure'], loc=2)
ax1.set(xlabel='Departure Time', ylabel='Occupancy')
ax2.plot(test_data.ActualDepartureTime,
         test_data.OccupancyCorrected, color='blue')
ax2.legend(['Planned departure'], loc=2)
ax2.set(xlabel='Departure Time', ylabel='Occupancy')
fig.suptitle('2022-02-11 \n trip number: {}'.format(trip_number))
plt.show()

# %%


deadhead_data = pd.read_csv(
    r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/BezettingFare.csv', sep=';')
# %%
