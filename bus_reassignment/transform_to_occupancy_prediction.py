from tokenize import group
from webbrowser import get
import pandas as pd
import numpy as np
import pyodbc
import datetime as dt


def connect_to_database():
    conn = pyodbc.connect('Driver={SQL Server Native Client 11.0};'
                          'Server=ZAKIR;'
                          'Database=keolis;'
                          'Trusted_Connection=yes;')
    cursor = conn.cursor()
    return cursor, conn


def export_occupancy_per_trip(date, line, direction, stop):
    timetable = pd.read_csv(
        r'path/', sep=';')

    data = pd.read_csv(
        r'path/'.format(data_name), sep=';')
    data = data.fillna(0)
    data['TripNumber'] = data['TripNumber'].astype('int64')
    data['Systeemlijnnr'] = data['Systeemlijnnr'].astype('int64')
    data['IdDimHalte'] = data['IdDimHalte'].astype('int64')
    data['OccupancyCorrected'] = data['OccupancyCorrected'].astype('float')
    data['ActualDepartureTime'] = data['ActualDepartureTime'].astype(
        'datetime64')
    cursor, conn = connect_to_database()
    column = ['OperatingDate', 'TripNumber', 'Direction', 'Systeemlijnnr',
              'ActualDepartureTime', 'IdDimHalte', 'OccupancyCorrected']
    data = data[column]
    data = data.values.tolist()
    sql_insert = '''
        declare @operating_date date = ?
        declare @trip_number bigint = ?
        declare @direction bigint = ?
        declare @system_linenr bigint = ?
        declare @dep_time datetime = ?
        declare @stop bigint = ?
        declare @occupancy float = ?

        
        INSERT INTO occupancy_per_trip (operating_date, trip_number, direction, system_linenr, dep_time, stop,  occupancy)
            VALUES (@operating_date, @trip_number, @direction, @system_linenr, @dep_time, @stop,  @occupancy)
        '''
    cursor.executemany(sql_insert, data)
    conn.commit()
    cursor.close()
    return print('Successful')


export_occupancy_per_trip(data_name='August22_30')


def lines_stops():
    stops_dict = {}
    cursor, conn = connect_to_database()
    data = pd.read_sql_query(
        'select * from occupancy_per_trip', conn)
    data = data.sort_values('dep_time')

    lines_list = [4701, 4702, 4703, 4704,
                  4705, 4706, 4707, 4708, 4709, 4060, 4061, 4062]
    data = data[data['system_linenr'].isin(lines_list)]

    data = data.drop_duplicates(
        subset=['system_linenr', 'direction', 'stop'], keep='first')

    stops_dict.update({i: list(k) for i, k in data.groupby(
        by=['system_linenr', 'direction'])['stop']})
    cursor.close()

    return lines_list, stops_dict


def occupancy_ratio():
    cursor, conn = connect_to_database()
    data = pd.read_sql(
        "SELECT * FROM occupancy_per_trip WHERE operating_date BETWEEN '2022-08-22' AND '2022-08-26'", conn)
    data['hour'] = pd.to_datetime(data['dep_time']).dt.hour
    cursor.close()

    grouped_data = pd.DataFrame(data.groupby(
        by=['operating_date', 'system_linenr', 'direction', 'stop', 'hour'])['trip_number'].count())
    grouped_data.reset_index(inplace=True)
    average_data = pd.DataFrame(grouped_data.groupby(
        by=['system_linenr', 'direction', 'stop', 'hour'])['trip_number'].mean()).reset_index()
    average_data['avg_trips'] = round(average_data['trip_number'])
    for i in range(len(data)):
        if data.loc[i, 'operating_date'] == (pd.to_datetime('2022-08-22')):

            for trip in data[data['operating_date'] == date]['trip_number']:
                print(date, trip)
