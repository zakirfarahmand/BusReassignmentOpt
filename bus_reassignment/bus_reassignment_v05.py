# import libraries
from typing import final
from soupsieve import select
import googlemaps
import gurobipy as gp
from gurobipy import GRB
from gurobipy import multidict, tuplelist, quicksum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time
from datetime import datetime
import random
import pyodbc
import datetime as dt
from datetime import timedelta, datetime
from itertools import combinations, product
import statistics
from matplotlib import animation
from IPython.display import HTML
from termcolor import colored
import googlemaps
import sys
import pytz

# 
# import functions
# insert year, month, and day


def connect_to_database():
    conn = pyodbc.connect('Driver={SQL Server Native Client 11.0};'
                          'Server=name;'
                          'Database=name;'
                          'Trusted_Connection=yes;')
    cursor = conn.cursor()
    return cursor, conn


def connect_to_databaseapi():
    conn = pyodbc.connect('Driver={SQL Server Native Client 11.0};'
                          'Server=name;'
                          'Database=name;'
                          'Trusted_Connection=yes;')
    cursor = conn.cursor()
    return cursor, conn


# list of trips exceeding the capacity threshold


def preliminary_parameters(date):
    date = str(date)
    cursor, conn = connect_to_database()
    data = pd.read_sql(
        "SELECT * FROM pred_occupancy_per_stop WHERE operating_date = '{}'".format(date), conn)
    timetable = pd.read_sql(
        "SELECT * FROM timetable WHERE operating_date = '{}'".format(date), conn)
    cursor.close()
    data = data.replace({np.nan: None})

    data = pd.merge(timetable, data, how='left', on=[
                    'trip_number', "system_linenr", 'direction', 'stop'])
    column = ['operating_date_x', 'system_linenr', 'direction', 'trip_number', 'stop',
              'passing_time_x', 'arrival_time_x', 'departure_time_x',
              'vehicle_number_x', 'prediction_per_hour',
              'occupancy_ratio', 'prediction_per_stop', 'prev_week_occupancy']
    data = data[column]

    for i in range(len(data)):
        if data.loc[i, 'passing_time_x'] is not None:
            data.loc[i, 'passing_time_x'] = data.loc[i, 'passing_time_x']
        else:
            data.loc[i, 'passing_time_x'] = data.loc[i, 'arrival_time_x']
    for i in range(len(data)):
        if data.loc[i, 'arrival_time_x'] is not None:
            data.loc[i, 'arrival_time_x'] = data.loc[i, 'arrival_time_x']
        else:
            data.loc[i, 'arrival_time_x'] = data.loc[i, 'passing_time_x']
    data.fillna(0, inplace=True)

    for i in range(len(data)):
        dep_time = data.loc[i, 'departure_time_x']
        data.loc[i, 'departure_time_x'] = dt.timedelta(
            hours=dep_time.hour, minutes=dep_time.minute, seconds=dep_time.second).total_seconds()

    for i in range(len(data)):
        pass_time = data.loc[i, 'passing_time_x']
        data.loc[i, 'passing_time_x'] = dt.timedelta(
            hours=pass_time.hour, minutes=pass_time.minute, seconds=pass_time.second).total_seconds()
    for i in range(len(data)):
        arr_time = data.loc[i, 'arrival_time_x']
        data.loc[i, 'arrival_time_x'] = dt.timedelta(
            hours=arr_time.hour, minutes=arr_time.minute, seconds=arr_time.second).total_seconds()

    first_stop_dict = {}
    last_stop_dict = {}
    bus_trips_dict = {}
    line_trips_dict = {}
    dep_time_dict = {}
    arr_time_dict = {}
    travel_time_dict = {}

    # create dictionary for the first stop of each trip
    first_stop = data.sort_values(by=['trip_number', 'passing_time_x']).drop_duplicates(
        subset=['trip_number'], keep='first')

    first_stop_dict.update({k: v for k, v in zip(
        first_stop['trip_number'], first_stop['stop'])})
    # create dictionary for the last stop of each trip
    last_stop = data.sort_values(by=['trip_number', 'passing_time_x']).drop_duplicates(
        subset=['trip_number'], keep='last')

    last_stop_dict.update({k: v for k, v in zip(
        last_stop['trip_number'], last_stop['stop'])})
    # create departure and arrival times dictionary
    dep_time_dict.update({i: j for i, j in zip(
        first_stop['trip_number'], first_stop['departure_time_x'])})
    arr_time_dict.update({i: j for i, j in zip(
        last_stop['trip_number'], last_stop['arrival_time_x'])})
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
                           for k, v in data.groupby('vehicle_number_x')['trip_number']})

    # return datasets
    return first_stop_dict, last_stop_dict, dep_time_dict, arr_time_dict, travel_time_dict


def calculate_waiting_time(date):

    waiting_time_dict = {}
    date = str(date)
    cursor, conn = connect_to_database()
    data = pd.read_sql(
        "SELECT * FROM timetable WHERE operating_date = '{}'".format(date), conn)
    cursor.close()
    data = data.replace({np.nan: None})

    # for i in range(len(data)):
    #     if data.loc[i, 'passing_time'] is not None:
    #         data.loc[i, 'hour'] = int(data.loc[i, 'passing_time']/3600)
    #     else:
    #         data.loc[i, 'hour'] = int(data.loc[i, 'arrival_time']/3600)

    for i in range(len(data)):
        if data.loc[i, 'passing_time'] is not None:
            data.loc[i, 'passing_time'] = data.loc[i, 'passing_time']
        else:
            data.loc[i, 'passing_time'] = data.loc[i, 'arrival_time']

    for i in range(len(data)):
        dep_time = data.loc[i, 'departure_time']
        data.loc[i, 'departure_time'] = dt.timedelta(
            hours=dep_time.hour, minutes=dep_time.minute, seconds=dep_time.second).total_seconds()

    for i in range(len(data)):
        pass_time = data.loc[i, 'passing_time']
        data.loc[i, 'passing_time'] = dt.timedelta(
            hours=pass_time.hour, minutes=pass_time.minute, seconds=pass_time.second).total_seconds()

    data = data.sort_values(
        by=['system_linenr', 'direction', 'passing_time'])

    # calculate headway for different time of the day: peak and off-peak
    data['headway'] = data.groupby(by=['system_linenr', 'direction', 'stop'])[
        'passing_time'].transform(pd.Series.diff)
    data = data.sort_values(
        by=['system_linenr', 'direction', 'stop', 'passing_time'])
    data['headway'].fillna(method='bfill', inplace=True)

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


def deadhead():
    cursor, conn = connect_to_database()
    deadhead_time = pd.read_sql_query('select * from deadhead_time', conn)
    deadhead_dict = {}

    cursor.close()
    deadhead_dict.update({(i, j): k for i, j, k in zip(
        deadhead_time.stopA, deadhead_time.stopB, deadhead_time.deadhead)})

    return deadhead_dict




def optimization_parameters(date):
    date = str(date)
    cursor, conn = connect_to_database()
    data = pd.read_sql(
        "SELECT * FROM pred_occupancy_per_stop WHERE operating_date = '{}'".format(date), conn)
    timetable = pd.read_sql(
        "SELECT * FROM timetable WHERE operating_date = '{}'".format(date), conn)
    cursor.close()
    data = data.replace({np.nan: None})

    data = pd.merge(timetable, data, how='left', on=[
                    'trip_number', "system_linenr", 'direction', 'stop'])
    column = ['operating_date_x', 'system_linenr', 'direction', 'trip_number', 'stop',
              'passing_time_x', 'arrival_time_x', 'departure_time_x',
              'vehicle_number_x', 'prediction_per_hour',
              'occupancy_ratio', 'prediction_per_stop', 'prev_week_occupancy']
    data = data[column]

    for i in range(len(data)):
        if data.loc[i, 'passing_time_x'] is not None:
            data.loc[i, 'passing_time_x'] = data.loc[i, 'passing_time_x']
        else:
            data.loc[i, 'passing_time_x'] = data.loc[i, 'arrival_time_x']
    for i in range(len(data)):
        if data.loc[i, 'arrival_time_x'] is not None:
            data.loc[i, 'arrival_time_x'] = data.loc[i, 'arrival_time_x']
        else:
            data.loc[i, 'arrival_time_x'] = data.loc[i, 'passing_time_x']
    data.fillna(0, inplace=True)

    for i in range(len(data)):
        dep_time = data.loc[i, 'departure_time_x']
        data.loc[i, 'departure_time_x'] = dt.timedelta(
            hours=dep_time.hour, minutes=dep_time.minute, seconds=dep_time.second).total_seconds()

    for i in range(len(data)):
        pass_time = data.loc[i, 'passing_time_x']
        data.loc[i, 'passing_time_x'] = dt.timedelta(
            hours=pass_time.hour, minutes=pass_time.minute, seconds=pass_time.second).total_seconds()

    first_trips = data.sort_values(by=['system_linenr', 'direction', 'departure_time_x']).drop_duplicates(
        subset=['system_linenr', 'direction', 'trip_number'], keep='first')
    first_trips = first_trips.drop_duplicates(
        subset=['system_linenr', 'direction'], keep='first')
    first_trips = [i for i in first_trips['trip_number']]
    last_trips = data.sort_values(by=['system_linenr', 'direction', 'departure_time_x']).drop_duplicates(
        subset=['system_linenr', 'direction', 'trip_number'], keep='last')
    last_trips = last_trips.drop_duplicates(
        subset=['system_linenr', 'direction'], keep='last')
    last_trips = [i for i in last_trips['trip_number']]
    data = data[~data['trip_number'].isin(first_trips)]
    data = data[~data['trip_number'].isin(last_trips)]
    data.dropna(inplace=True)
    # remove the first and last trip of the day
    capacity_threshold = 60  # capacity threshold
    occupancy_dict = {}  # number of expected in-vehicle passengers
    stranded_pass_dict = {}  # number of passengers exceeding the capacity threshold
    stops_dict = {}  # list of all stops along each trip
    trip_following_trips = {}  # list of following trips
    bus_trips = {}  # list trips operated by the same bus
    line_trips = {}  # list of trips on the same bus line
    toAssign = []
    reAssign = []
    # calculate occupancy exceeding the capacity threshold
    data['stranded_passengers'] = data['prediction_per_stop'].apply(
        lambda x: x - capacity_threshold if (x > capacity_threshold) else 0)

    occupancy_dict.update({(i, j): k for i, j, k in zip(
        data['trip_number'], data['stop'], data['prediction_per_stop'])})

    stranded_pass_dict.update({(i, j): k for i, j, k in zip(
        data['trip_number'], data['stop'], data['stranded_passengers'])})

    stops_dict = {k: list(v)
                  for k, v in data.groupby('trip_number')['stop']}

    enschede_lines = [4701, 4702, 4703, 4704, 4705, 4706, 4707, 4709]

    enschede = data[data['system_linenr'].isin(enschede_lines)]
    # list of trips exceeding the capacity threshold
    toAssign += [
        k for k in enschede[enschede['stranded_passengers'] > 0]['trip_number'].drop_duplicates(keep='first')]

    # list of trips that could potentially be re-assigned
    reAssign = enschede[~enschede['trip_number'].isin(toAssign)]
    line9 = [4709]    # any trips on line 9 cannot be reassigned
    reAssign = reAssign[~reAssign['system_linenr'].isin(line9)]
    reAssign = reAssign['trip_number'].drop_duplicates(keep='first').tolist()

    # list of three following trips for each trip operated by the same bus

    sorted_data = data.sort_values(
        by=['vehicle_number_x', 'departure_time_x']).drop_duplicates(subset=['trip_number'], keep='first')

    bus_trips.update({k: list(v)
                      for k, v in sorted_data.groupby('vehicle_number_x')['trip_number']})

    all_following_trips = {}
    for trip in reAssign:
        list_ftrips = []
        for key, value in bus_trips.items():
            if trip in value:
                all_following_trips.update({trip: value})

    dep_time = data.sort_values(by=['system_linenr', 'departure_time_x']).drop_duplicates(
        subset=['trip_number'], keep='first')
    dep_time_dict = {k: v for k, v in zip(
        dep_time['trip_number'], dep_time['departure_time_x'])}

    for key, value in all_following_trips.items():
        following_trips_list = []
        for i in value:
            if dep_time_dict[key] < dep_time_dict[i] and i != key:
                following_trips_list += [i]
        all_following_trips.update({key: following_trips_list[0:4]})
    # remove trips which has no following trips
    for key, value in all_following_trips.items():
        if len(value) >= 2:
            trip_following_trips.update({key: value[0:4]})
    # precondition 1: a trip cannot be reassigned if the following trips running by the same bus is expected to be overcrowded
    for key, value in trip_following_trips.items():
        list_following_trips = []
        for i in value:
            if i not in toAssign:
                list_following_trips += [i]
            trip_following_trips.update({key: list_following_trips})
    # update the reassign list
    reAssign = [x for x in trip_following_trips.keys()]

    # list of preceeding trips on the same bus line
    sorted_data2 = data.sort_values(
        by=['system_linenr', 'departure_time_x']).drop_duplicates(subset=['trip_number'], keep='first')

    line_trips.update({k: list(v)
                       for k, v in sorted_data2.groupby(['system_linenr', 'direction'])['trip_number']})

    all_preceeding_trips = {}
    for trip in toAssign:
        for key, value in line_trips.items():
            if trip in value:
                all_preceeding_trips.update({trip: value})
    preceeding_trip = {}
    for key, value in all_preceeding_trips.items():
        preceeding_list = []
        for i in value:
            if dep_time_dict[i] < dep_time_dict[key]:
                preceeding_list += [i]
                preceeding_list.reverse()
        preceeding_trip.update({key: preceeding_list[0]})
    reAssign = [a for a in reAssign if a not in preceeding_trip.values()]

    return occupancy_dict, stranded_pass_dict, stops_dict, toAssign, reAssign, trip_following_trips, preceeding_trip


'''
For creating list B, these preconditions should met:
   1. trips should depart before depature time of trips in A
   2. their following trips should not be overcrowded
'''


def bus_reassginment(date):
    time_window = 600

    deadhead_dict = deadhead()
    occupancy_dict, stranded_pass_dict, stops_dict, toAssign, reAssign, trip_following_trips, preceeding_trip = optimization_parameters(
        date)

    waiting_time_dict = calculate_waiting_time(date)

    first_stop_dict, last_stop_dict, dep_time_dict, arr_time_dict, travel_time_dict = preliminary_parameters(
        date)

    model = gp.Model('Bus Reassignment')
    epsilon = 120  # this is the time for boarding passengers

    bigM = 1.64456466e+12  # big value
    deadhead_threshold = 900  # deadhead time threshold is set 15 minutes
    # create pair of potential trips for re-assignemnt and trips that they could be re-assigned before
    if len(toAssign) > 0:
        pass
    else:
        print(colored("Good News! No overcrowded trip was found",
              'green', attrs=['bold']))
        sys.exit()

    paired_trips = tuplelist()
    for i in reAssign:
        for j in toAssign:
            if (dep_time_dict[i] + deadhead_dict[(first_stop_dict[i], first_stop_dict[j])] <= dep_time_dict[j] + time_window) & (dep_time_dict[i] + deadhead_dict[(first_stop_dict[i], first_stop_dict[j])] >= dep_time_dict[j] - time_window):
                paired_trips += [(i, j)]
    # x = model.addVars(trip_pairs, vtype=GRB.BINARY, name='x')
    reassign_var = model.addVars(
        paired_trips, vtype=GRB.BINARY, name="x[%s, %s]" % (i, j))
    # create pair of potential imposedcancellations
    imposed_cancellation = tuplelist()
    for i, j in paired_trips:
        for k in trip_following_trips[i]:
            imposed_cancellation += [(i, j, k)]

    imposed_var = model.addVars(
        imposed_cancellation, vtype=GRB.BINARY, name="x[%s, %s, %s]" % (i, j, k))
    model.update()
    # objective
    obj = quicksum(0.5 * reassign_var[i, j] * stranded_pass_dict[j, s] * waiting_time_dict[j, s] for i, j in paired_trips for s in stops_dict[j]) + quicksum(3 * (1-reassign_var[i, j]) * stranded_pass_dict[j, s] * waiting_time_dict[j, s] for i, j in paired_trips for s in stops_dict[j]) + quicksum(2 * reassign_var[i, j] * occupancy_dict[i, s]
                                                                                                                                                                                                                                                                                                         * waiting_time_dict[i, s] for i, j in paired_trips for s in stops_dict[i]) + quicksum(2 * imposed_var[i, j, k] * occupancy_dict[k, s] * waiting_time_dict[k, s] for i, j in paired_trips for k in trip_following_trips[i] for s in stops_dict[k])  # + quicksum(0.001 * imposed_var[i,j,k] for i, j, k in imposed_cancellation)
    model.setObjective(obj, GRB.MINIMIZE)
    model.update()
    # add constraints
    # assign only one trip
    model.addConstrs((reassign_var.sum('*', j) ==
                     1 for j in toAssign), name="toAssign[%s]" % j)
    # cancell a trip only once
    model.addConstrs((reassign_var.sum(i, '*') <=
                     1 for i in reAssign), name='cancellation[%s]' % i)
    # deadhead time should not exceed the threshold
    model.addConstrs((reassign_var[i, j] * deadhead_dict[(first_stop_dict[i], first_stop_dict[j])]
                     <= deadhead_threshold for i, j in paired_trips), name='deadhead[%s, %s]' % (i, j))
    # lastest departure time of re-assigned trips
    # model.addConstrs((reassign_var[i, j] * (dep_time_dict[i] + deadhead_dict[(first_stop_dict[i], first_stop_dict[j])])
    #                  <= dep_time_dict[j] + time_window for i, j in paired_trips), name='departureTimeUp[%s, %s]' % (i, j))
    # model.addConstrs((- reassign_var[i, j] * (dep_time_dict[i] + deadhead_dict[(first_stop_dict[i], first_stop_dict[j])])
    #                  <= time_window - dep_time_dict[j] for i, j in paired_trips), name='departureTimeUp[%s, %s]' % (i, j))
    # earliest departure time of re-assigned trips: before departure of the very first preceeding trip
    model.addConstrs((reassign_var[i, j] * (dep_time_dict[preceeding_trip[j]]) <= dep_time_dict[i] + deadhead_dict[(first_stop_dict[i], first_stop_dict[j])]
                      for i, j in paired_trips), name='departureTimeUp[%s, %s]' % (i, j))
    model.addConstrs((- reassign_var[i, j] + imposed_var[i, j, k] <=
                     0 for i, j in paired_trips for k in trip_following_trips[i]))
    # imposed cancellation
    for i, j in paired_trips:
        for p in trip_following_trips[i]:
            model.addConstr((reassign_var[i, j] * (dep_time_dict[i] + deadhead_dict[(first_stop_dict[i], first_stop_dict[j])] + travel_time_dict[j] + deadhead_dict[(
                last_stop_dict[j], first_stop_dict[p])])) - bigM * imposed_var[i, j, p] <= dep_time_dict[p])

        # max number of imposed cancellation
    model.addConstrs((imposed_var.sum(i, j, '*'))
                     <= 1 for i, j in paired_trips)
    model.update()

    model.optimize()
    model.__data1 = reassign_var
    model.__data2 = imposed_var
    active_arcs = [a for a in paired_trips if model.__data1[a].x > 0.99]

    print(colored("Optimal solutions:", 'green', attrs=['bold']))
    for i in active_arcs:
        print(
            "Trip {} can be re-assigned".format(i[0]) + ' ' + "before trip {}".format(i[1]))
    for j in toAssign:
        if j not in [a[1] for a in active_arcs]:
            print("No optimal bus trip was found to re-assign before trip {}".format(j))
    print(colored("Imposed Cancellations:", 'green', attrs=['bold']))
    imposed_arc = [
        a for a in imposed_cancellation if model.__data2[a].x > 0.99]
    for i in imposed_arc:
        print(
            "Trip {}".format(i[2]) + ' ' + " is cancelled as a result of cancelling trip {}".format(i[0]) + ' ' + 'and re-assigned before trip {}'.format(i[1]))
    for i in active_arcs:
        if i[0] not in [a[0] for a in imposed_arc]:
            print(
                "No imposed cancellation(s) due to re-assigning trip {}".format(i[0]) + ' ' + 'before trip {}'.format(i[1]))
    return active_arcs, imposed_arc, toAssign, first_stop_dict, dep_time_dict, trip_following_trips, deadhead_dict


# active_arcs, imposed_arc = bus_reassginment('2022-09-26')


date = dt.datetime.today().date() - timedelta(days=6)


def export_optimization_results(date):
    active_arcs, imposed_arc, toAssign, first_stop_dict, dep_time_dict, trip_following_trips, deadhead_dict = bus_reassginment(
        date)

    date = str(date)
    cursor, conn = connect_to_database()
    data = pd.read_sql(
        "SELECT * FROM pred_occupancy_per_stop WHERE operating_date = '{}'".format(date), conn)
    timetable = pd.read_sql(
        "SELECT * FROM timetable WHERE operating_date = '{}'".format(date), conn)
    cursor.close()
    stops = pd.read_csv(
        r'C:/Users/FARAHMANDZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/bus_stops.csv', sep=';')
    data = pd.merge(data, stops, left_on=['stop'], right_on=['IdDimHalte'])

    data = data.replace({np.nan: None})

    data = pd.merge(timetable, data, how='left', on=[
                    'trip_number', "system_linenr", 'direction', 'stop'])

    column = ['operating_date_x', 'system_linenr', 'direction', 'trip_number', 'stop',
              'passing_time_x', 'arrival_time_x', 'departure_time_x',
              'vehicle_number_x', 'prediction_per_hour',
              'occupancy_ratio', 'prediction_per_stop', 'prev_week_occupancy']
    data = data[column]

    data = pd.merge(data, stops, left_on=['stop'], right_on=['IdDimHalte'])

    for i in range(len(data)):
        if data.loc[i, 'passing_time_x'] is not None:
            data.loc[i, 'passing_time_x'] = data.loc[i, 'passing_time_x']
        else:
            data.loc[i, 'passing_time_x'] = data.loc[i, 'arrival_time_x']
    for i in range(len(data)):
        if data.loc[i, 'arrival_time_x'] is not None:
            data.loc[i, 'arrival_time_x'] = data.loc[i, 'arrival_time_x']
        else:
            data.loc[i, 'arrival_time_x'] = data.loc[i, 'passing_time_x']
    data.fillna(0, inplace=True)
    trips = data.sort_values(by=['trip_number', 'passing_time_x']).drop_duplicates(
        subset=['trip_number'], keep='first')
    trip_line_direction = {k: (v, j) for k, v, j in zip(
        trips['trip_number'], trips['system_linenr'], trips['direction'])}
    trip_vehicle = {k: v for k, v in zip(
        trips['trip_number'], trips['vehicle_number_x'])}
    # list of overcrowded trips under the pilot
    df1 = data[data['trip_number'].isin(toAssign)]
    most_crowded_stop = {}
    for trip in toAssign:
        most_crowded_stop.update(
            {trip: df1[df1['trip_number'] == trip]['prediction_per_stop'].max()})
    df1 = df1.sort_values(by=['passing_time_x']).drop_duplicates(
        'trip_number', keep='first')
    df1.reset_index(inplace=True)
    df1['most_crowded_stop'] = ''
    for i in range(len(df1)):
        for key in most_crowded_stop.keys():
            if df1.loc[i, 'trip_number'] == key:
                df1.loc[i, 'most_crowded_stop'] = most_crowded_stop[key]
    df1 = df1.rename(columns={'operating_date_x': 'operating_date', 'system_linenr': "system_linenr_toAssign", 'direction': 'direction_toAssign',
                              'trip_number': 'toAssign', 'Naam_halte': 'start_stop_toAssign', 'departure_time_x': "departure_toAssign", 'vehicle_number_x': "vehicle_toAssign"})
    df1 = df1[['toAssign', 'operating_date', 'system_linenr_toAssign',
               'direction_toAssign', 'start_stop_toAssign', 'departure_toAssign', 'most_crowded_stop',
               'vehicle_toAssign']]
    # list of reassigned trips
    df2 = pd.DataFrame(active_arcs, columns=['reAssign', 'toAssign'])
    df2 = pd.merge(df2, trips, how='left', left_on=[
                   'reAssign'], right_on=['trip_number'])
    df2 = df2.rename(columns={'operating_date_x': 'operating_date', 'system_linenr': 'system_linenr_reAssign',
                              'direction': 'direction_reAssign', 'Naam_halte': 'start_stop_reAssign', 'departure_time_x': 'departure_reAssign', 'vehicle_number_x': 'vehicle_reAssign'})
    df2 = df2[['toAssign', 'reAssign', 'operating_date', 'system_linenr_reAssign',
               'direction_reAssign',
               'start_stop_reAssign', 'departure_reAssign', 'vehicle_reAssign']]
    # list of imposed cancelled trips
    df3 = pd.DataFrame(imposed_arc, columns=[
                       'reAssign', 'toAssign', 'Imposed'])
    df3 = pd.merge(df3, trips, how='left', left_on=[
                   'Imposed'], right_on=['trip_number'])
    df3 = df3.rename(columns={'operating_date_x': 'operating_date', 'system_linenr': 'system_linenr_imposed',
                              'direction': 'direction_imposed', 'Naam_halte': 'start_stop_imposed', 'departure_time_x': 'departure_imposed', 'vehicle_number_x': 'vehicle_imposed'})
    df3 = df3[['toAssign', 'reAssign', 'Imposed', 'operating_date',
               'system_linenr_imposed', 'direction_imposed', 'start_stop_imposed', 'departure_imposed',
               'vehicle_imposed']]

    final_data = pd.merge(df1, df2, how='left', on=[
                          'toAssign', 'operating_date'])
    final_data = pd.merge(final_data, df3, how='left', on=[
                          'toAssign', 'reAssign', 'operating_date'])

    final_data = final_data.replace({np.nan: None})
    final_data['assign_from_depot'] = ''
    for i in range(len(final_data)):
        if final_data.loc[i, 'reAssign'] is None:
            final_data.loc[i,
                           'assign_from_depot'] = 'Potential assignment from depot'
    final_data['deadhead'] = ''
    for i in range(len(final_data)):
        if final_data.loc[i, 'reAssign'] is not None:
            deadhead = (deadhead_dict[(
                first_stop_dict[final_data.loc[i, 'reAssign']], first_stop_dict[final_data.loc[i, 'toAssign']])])/60
            final_data.loc[i, 'deadhead'] = round(deadhead)
    final_data['departure_toAssign'] = final_data['departure_toAssign'].astype(
        'str')
    final_data['operating_date'] = final_data['operating_date'].astype('str')
    final_data['date_time'] = final_data['operating_date'] + \
        ' ' + final_data['departure_toAssign']
    final_data['date_time'] = pd.to_datetime(
        final_data['date_time'], infer_datetime_format=True)

    column = ['operating_date', 'date_time', 'toAssign', 'system_linenr_toAssign',
              'direction_toAssign', 'start_stop_toAssign', 'departure_toAssign',
              'most_crowded_stop', 'vehicle_toAssign', 'assign_from_depot', 'reAssign',
              'system_linenr_reAssign', 'direction_reAssign', 'start_stop_reAssign',
              'departure_reAssign', 'vehicle_reAssign', 'deadhead', 'Imposed'
              ]
    final_data = final_data[column]
    final_data = final_data.values.tolist()

    cursor, conn = connect_to_databaseapi()
    sql_insert = '''
        declare @operating_date date = ?
        declare @date_time datetime = ?
        declare @toAssign bigint = ?
        declare @system_linenr_toAssign bigint = ?
        declare @direction_toAssign bigint = ?
        declare @start_stop_toAssign nvarchar(50) = ?
        declare @departure_toAssign time = ?
        declare @most_overcrowded  float = ?
        declare @vehicle_toAssign bigint = ?
        declare @reassigned_from_depot nvarchar(50) = ?
        declare @reAssign bigint = ?
        declare @system_linenr_reAssign bigint = ?
        declare @direction_reAssign bigint = ?
        declare @start_stop_reAssign nvarchar(50) = ?
        declare @departure_reAssign time = ?
        declare @vehicle_reAssign bigint = ?
        declare @deadhead bigint = ?
        declare @imposed_cancellation bigint = ?

        UPDATE api_reassignedtrips
        SET date_time=@date_time, system_linenr_toAssign=@system_linenr_toAssign, direction_toAssign=@direction_toAssign, start_stop_toAssign=@start_stop_toAssign, departure_toAssign=@departure_toAssign,
            most_overcrowded=@most_overcrowded, vehicle_toAssign=@vehicle_toAssign,reassigned_from_depot=@reassigned_from_depot, reAssign=@reAssign,  system_linenr_reAssign=@system_linenr_reAssign, direction_reAssign=@direction_reAssign, start_stop_reAssign=@start_stop_reAssign,
            departure_reAssign=@departure_reAssign, vehicle_reAssign=@vehicle_reAssign, deadhead=@deadhead, imposed_cancellation=@imposed_cancellation
        WHERE operating_date = @operating_date AND toAssign = @toAssign

        IF @@ROWCOUNT = 0
            INSERT INTO api_reassignedtrips
            (operating_date, date_time, toAssign, system_linenr_toAssign,direction_toAssign, start_stop_toAssign, departure_toAssign,
            most_overcrowded, vehicle_toAssign,reassigned_from_depot, reAssign, system_linenr_reAssign, direction_reAssign, start_stop_reAssign,
            departure_reAssign, vehicle_reAssign, deadhead, imposed_cancellation)
            VALUES (@operating_date, @date_time, @toAssign, @system_linenr_toAssign, @direction_toAssign, @start_stop_toAssign, @departure_toAssign,
            @most_overcrowded, @vehicle_toAssign,  @reassigned_from_depot, @reAssign, @system_linenr_reAssign, @direction_reAssign, @start_stop_reAssign,
            @departure_reAssign, @vehicle_reAssign, @deadhead, @imposed_cancellation)
        '''
    cursor.executemany(sql_insert, final_data)

    conn.commit()
    cursor.close()

