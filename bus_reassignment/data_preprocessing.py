from math import comb
import statistics
import datetime as dt
from datetime import datetime, timedelta

from traceback import format_exception
from unittest.util import sorted_list_difference
from webbrowser import get
from click import secho
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gurobipy import tuplelist
import googlemaps
from soupsieve import select
import pyodbc
from matplotlib import style
plt.style.use('seaborn-dark')

# Requires API key
gmaps = googlemaps.Client(key='AIzaSyAUksLuaZra4mCcOLZL_52b9nIvHa7TgFw')
# gmaps = googlemaps.Client(key='AIzaSyAra0o3L3rs-uHn4EpaXx1Y57SIF_02684')


def connect_to_database():
    conn = pyodbc.connect('Driver={SQL Server Native Client 11.0};'
                          'Server=UT163156;'
                          'Database=keolis;'
                          'Trusted_Connection=yes;')
    cursor = conn.cursor()
    return cursor, conn




def import_data(date):
    # call data from the database
    cursor, conn = connect_to_database()
    data = pd.read_sql_query(
        "select * from pred_occupancy_per_stop where operating_date = '{}' ".format(date), conn)
    cursor.close()

    data['departure_time'] = pd.to_datetime(
        data['departure_time'], format='%Y-%m-%d %H:%M:%S')
    data['passing_time'] = pd.to_datetime(
        data['passing_time'], format='%Y-%m-%d %H:%M:%S')

    data.sort_values(by=['dep_time', 'system_linenr',
                     'direction', 'trip_number'], inplace=True)
    return data


def preliminary_parameters(date):
    cursor, conn = connect_to_database()
    data = pd.read_sql(
        "SELECT * FROM trips_timetable WHERE operating_date = '{}'".format(date), conn)
    cursor.close()

    first_stop_dict = {}
    last_stop_dict = {}
    bus_trips_dict = {}
    line_trips_dict = {}
    dep_time_dict = {}
    arr_time_dict = {}
    travel_time_dict = {}
    # convert time to seconds
    for i in range(len(data)):
        dep_time = data.loc[i, 'departure_time']
        data.loc[i, 'departure_time'] = dt.timedelta(
            hours=dep_time.hour, minutes=dep_time.minute, seconds=dep_time.second).total_seconds()
    for i in range(len(data)):
        arr_time = data.loc[i, 'arrival_time']
        data.loc[i, 'arrival_time'] = dt.timedelta(
            hours=arr_time.hour, minutes=arr_time.minute, seconds=arr_time.second).total_seconds()
    # create dictionary for the first stop of each trip
    first_stop_dict.update({k: v for k, v in zip(
        data['trip_number'], data['start_stop'])})
    # create dictionary for the last stop of each trip
    last_stop_dict.update({k: v for k, v in zip(
        data['trip_number'], data['last_stop'])})
    # create departure and arrival times dictionary
    dep_time_dict.update({i: j for i, j in zip(
        data['trip_number'], data['departure_time'])})
    arr_time_dict.update({i: j for i, j in zip(
        data['trip_number'], data['arrival_time'])})
    # calculate travel time for each trip
    for k1, v1 in dep_time_dict.items():
        for k2, v2 in arr_time_dict.items():
            if k1 == k2:
                travel_time = v2 - v1
                travel_time_dict.update({k1: travel_time})
    # list of trips on the same line
    line_trips_dict.update({k: list(v) for k, v in data.groupby(
        'system_linenr')['trip_number']})
    # list of trips operating by the same bus
    bus_trips_dict.update({k: list(v)
                           for k, v in data.groupby('vehicle_number')['trip_number']})

    # return datasets
    return first_stop_dict, last_stop_dict, dep_time_dict, arr_time_dict, travel_time_dict


def calculate_waiting_time(date):

    waiting_time_dict = {}

    cursor, conn = connect_to_database()
    data = pd.read_sql(
        "SELECT * FROM timetable WHERE operating_date = '{}'".format(date), conn)
    cursor.close()

    # convert time to seconds
    for i in range(len(data)):
        dep_time = data.loc[i, 'departure_time']
        data.loc[i, 'departure_time'] = dt.timedelta(
            hours=dep_time.hour, minutes=dep_time.minute, seconds=dep_time.second).total_seconds()
    for i in range(len(data)):
        pass_time = data.loc[i, 'passing_time']
        data.loc[i, 'passing_time'] = dt.timedelta(
            hours=pass_time.hour, minutes=pass_time.minute, seconds=pass_time.second).total_seconds()

    for i in range(len(data)):
        data.loc[i, 'hour'] = int(data.loc[i, 'passing_time']/3600)

    data = data.sort_values(by=['system_linenr', 'direction', 'passing_time'])

    # calculate headway for different time of the day: peak and off-peak
    data['headway'] = data.groupby(by=['system_linenr', 'direction', 'stop'])[
        'passing_time'].transform(pd.Series.diff)
    data.dropna(inplace=True)

    data['variance'] = data.groupby(by=['system_linenr', 'direction', 'stop'])[
        'headway'].transform(statistics.variance)
    data['mean'] = data.groupby(by=['system_linenr', 'direction', 'stop'])[
        'headway'].transform(statistics.mean)

    def cal_waiting_time(mean, var):
        waiting_time = mean * 0.5 + 0.5 * (var / mean)
        return waiting_time
    # calculate waiting time
    data['waiting_time'] = cal_waiting_time(
        data['mean'], data['variance'])

    waiting_time_dict.update({(i, j): k for i, j, k in zip(
        data['trip_number'], data['stop'], data['waiting_time'])})

    return waiting_time_dict


def calculate_distance(lat1, lon1, lat2, lon2):

    distance = gmaps.distance_matrix([str(lat1) + " " + str(lon1)],
                                     [str(lat2) + " " + str(lon2)],
                                     departure_time=datetime.now().timestamp(),
                                     mode='driving')["rows"][0]["elements"][0]["duration"]["value"]
    return distance


def calculate_deadhead(date):
    cursor, conn = connect_to_database()
    date= str(date)
    data = pd.read_sql(
        "SELECT * FROM timetable WHERE operating_date = '{}'".format(date), conn)
    data = data.replace({np.nan: None})

    for i in range(len(data)):
        if data.loc[i, 'passing_time'] is not None:
            data.loc[i, 'passing_time'] = data.loc[i, 'passing_time']
        else:
            data.loc[i, 'passing_time'] = data.loc[i, 'arrival_time']
    first_stop = data.sort_values(by=['trip_number', 'passing_time']).drop_duplicates(
        subset=['trip_number'], keep='first')

    # create dictionary for the last stop of each trip
    last_stop = data.sort_values(by=['trip_number', 'passing_time']).drop_duplicates(
        subset=['trip_number'], keep='last')

    stops = pd.read_csv(
        r'C:/Users/FARAHMANDZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/bus_stops.csv', sep=';')
    cursor.close()

    first_stop = pd.merge(first_stop, stops, left_on=[
                          'stop'], right_on=['IdDimHalte'])
    last_stop = pd.merge(last_stop, stops, left_on=[
        'stop'], right_on=['IdDimHalte'])

    first_stop.drop_duplicates('stop', inplace=True, keep='first')

    last_stop.drop_duplicates('stop', inplace=True, keep='first')
    first_stop_dict = {}
    last_stop_dict = {}
    first_stop_dict.update({i: (j, v) for i, j, v in zip(
        first_stop['stop'],  first_stop['Breedtegraad'], first_stop['Lengtegraad'])})
    # create dictionary for the last stop of each trip
    last_stop_dict.update({i: (j, v) for i, j, v in zip(
        last_stop['stop'], last_stop['Breedtegraad'], last_stop['Lengtegraad'])})

    data = pd.DataFrame(columns=['stopA', 'stopB', 'deadhead'])
    for key1, val1 in first_stop_dict.items():
        for key2, val2 in first_stop_dict.items():
            if key1 != key2:
                data = data.append({'stopA': key1, 'stopB': key2, 'deadhead': calculate_distance(
                    val1[0], val1[1], val2[0], val2[1])}, ignore_index=True)
            elif key1 == key2:
                data = data.append(
                    {'stopA': key1, 'stopB': key2, 'deadhead': 0}, ignore_index=True)
    for key1, val1 in first_stop_dict.items():
        for key2, val2 in last_stop_dict.items():
            if key1 != key2:
                data = data.append({'stopA': key1, 'stopB': key2, 'deadhead': calculate_distance(
                    val1[0], val1[1], val2[0], val2[1])}, ignore_index=True)
            elif key1 == key2:
                data = data.append(
                    {'stopA': key1, 'stopB': key2, 'deadhead': 0}, ignore_index=True)
    for key1, val1 in last_stop_dict.items():
        for key2, val2 in first_stop_dict.items():
            if key1 != key2:
                data = data.append({'stopA': key1, 'stopB': key2, 'deadhead': calculate_distance(
                    val1[0], val1[1], val2[0], val2[1])}, ignore_index=True)
            elif key1 == key2:
                data = data.append(
                    {'stopA': key1, 'stopB': key2, 'deadhead': 0}, ignore_index=True)

    data.drop_duplicates(subset=['stopA', 'stopB'], inplace=True, keep='first')
    # directly store the data to the databse
    data = data.values.tolist()
    cursor, conn = connect_to_database()
    sql_insert = '''
        declare @stopA bigint = ?
        declare @stopB bigint = ?
        declare @deadhead float = ?

        UPDATE deadhead_time    
        SET deadhead = @deadhead
        WHERE stopA = @stopA AND stopB = @stopB

        IF @@ROWCOUNT = 0
            INSERT INTO deadhead_time
                (stopA, stopB, deadhead)
            VALUES (@stopA, @stopB, @deadhead)
        '''
    cursor.executemany(sql_insert, data)

    conn.commit()

date = dt.datetime.today().date() + timedelta(days=1)
calculate_deadhead(date)
