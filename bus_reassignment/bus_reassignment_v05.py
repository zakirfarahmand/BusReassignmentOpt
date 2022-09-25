# import libraries
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

# AIzaSyBmoRMRmfpYO5AG3GsSbmKlyNPaWOkKROE
# import functions
# insert year, month, and day


def connect_to_database():
    conn = pyodbc.connect('Driver={SQL Server Native Client 11.0};'
                          'Server=ZAKIR;'
                          'Database=keolis;'
                          'Trusted_Connection=yes;')
    cursor = conn.cursor()
    return cursor, conn


def connect_to_databaseapi():
    conn = pyodbc.connect('Driver={SQL Server Native Client 11.0};'
                          'Server=ZAKIR;'
                          'Database=keodss3.0;'
                          'Trusted_Connection=yes;')
    cursor = conn.cursor()
    return cursor, conn

date = dt.datetime.today().date()



    # list of trips exceeding the capacity threshold
   
    
def preliminary_parameters(date):
    cursor, conn = connect_to_database()
    data = pd.read_sql(
        "SELECT * FROM pred_occupancy_per_stop WHERE operating_date = '{}'".format(date), conn)
    cursor.close()
    data = data.replace({np.nan: None})

    for i in range(len(data)):
        if data.loc[i, 'passing_time'] is not None:
            data.loc[i, 'passing_time'] = data.loc[i, 'passing_time']
        else:
            data.loc[i, 'passing_time'] = data.loc[i, 'arrival_time']
    first_stop_dict = {}
    last_stop_dict = {}
    bus_trips_dict = {}
    line_trips_dict = {}
    dep_time_dict = {}
    arr_time_dict = {}
    travel_time_dict = {}

    # create dictionary for the first stop of each trip
    first_stop = data.sort_values(by=['trip_number', 'passing_time']).drop_duplicates(
        subset=['trip_number'], keep='first')

    first_stop_dict.update({k: v for k, v in zip(
        first_stop['trip_number'], first_stop['stop'])})
    # create dictionary for the last stop of each trip
    last_stop = data.sort_values(by=['trip_number', 'passing_time']).drop_duplicates(
        subset=['trip_number'], keep='last')

    last_stop_dict.update({k: v for k, v in zip(
        last_stop['trip_number'], last_stop['stop'])})
    # create departure and arrival times dictionary
    dep_time_dict.update({i: j for i, j in zip(
        first_stop['trip_number'], first_stop['departure_time'])})
    arr_time_dict.update({i: j for i, j in zip(
        last_stop['trip_number'], last_stop['arrival_time'])})
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
        "SELECT * FROM pred_occupancy_per_stop WHERE operating_date = '{}'".format(date), conn)
    cursor.close()
    data = data.replace({np.nan: None})

    for i in range(len(data)):
        if data.loc[i, 'passing_time'] is not None:
            data.loc[i, 'hour'] = int(data.loc[i, 'passing_time']/3600)
        else:
            data.loc[i, 'hour'] = int(data.loc[i, 'arrival_time']/3600)

    for i in range(len(data)):
        if data.loc[i, 'passing_time'] is not None:
            data.loc[i, 'passing_time'] = data.loc[i, 'passing_time']
        else:
            data.loc[i, 'passing_time'] = data.loc[i, 'arrival_time']

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


# def pred_occupancy_data(date):
#     # call data from the database
#     cursor, conn = connect_to_database()
#     data = pd.read_sql_query(
#         "select * from pred_occupancy_per_stop where operating_date = '{}' ".format(date), conn)
#     cursor.close()


#     # fix data format
#     # data['ActualArrivalTime'] = pd.to_datetime(
#     #     data['ActualArrivalTime'], format='%Y-%m-%d %H:%M:%S')
#     data['ActualDepartureTime'] = pd.to_datetime(
#         data['ActualDepartureTime'], format='%Y-%m-%d %H:%M:%S')
#     # data['ArrivalTime'] = pd.to_datetime(
#     #     data['ArrivalTime'], format='%Y-%m-%d %H:%M:%S')
#     data['DepartureTime'] = pd.to_datetime(
#         data['DepartureTime'], format='%Y-%m-%d %H:%M:%S')
#     # fill nan values
#     data.sort_values(by=['TripNumber', 'DepartureTime',
#                          'Systeemlijnnr'], inplace=True)

#     return data


def deadhead():
    cursor, conn = connect_to_database()
    deadhead_time = pd.read_sql_query('select * from deadhead_time', conn)
    deadhead_dict = {}

    cursor.close()
    deadhead_dict.update({(i, j): k for i, j, k in zip(
        deadhead_time.stopA, deadhead_time.stopB, deadhead_time.deadhead)})

    return deadhead_dict


# %%


''' list of to assign and re-assign should only from Enschede bus lines '''


def optimization_parameters(date):
    cursor, conn = connect_to_database()
    data = pd.read_sql_query(
        "select * from pred_occupancy_per_stop where operating_date = '{}' ".format(date), conn)
    cursor.close()
    first_trips = data.sort_values(by=['system_linenr', 'direction', 'departure_time']).drop_duplicates(
        subset=['system_linenr', 'direction', 'trip_number'], keep='first')
    first_trips = first_trips.drop_duplicates(
        subset=['system_linenr', 'direction'], keep='first')
    first_trips = [i for i in first_trips['trip_number']]
    last_trips = data.sort_values(by=['system_linenr', 'direction', 'departure_time']).drop_duplicates(
        subset=['system_linenr', 'direction', 'trip_number'], keep='last')
    last_trips = last_trips.drop_duplicates(
        subset=['system_linenr', 'direction'], keep='last')
    last_trips = [i for i in last_trips['trip_number']]
    data = data[~data['trip_number'].isin(first_trips)]
    data = data[~data['trip_number'].isin(last_trips)]
    data.dropna(inplace=True)
    # remove the first and last trip of the day
    capacity_threshold = 50  # capacity threshold
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
        by=['vehicle_number', 'departure_time']).drop_duplicates(subset=['trip_number'], keep='first')

    bus_trips.update({k: list(v)
                      for k, v in sorted_data.groupby('vehicle_number')['trip_number']})

    all_following_trips = {}
    for trip in reAssign:
        list_ftrips = []
        for key, value in bus_trips.items():
            if trip in value:
                all_following_trips.update({trip: value})

    dep_time = data.sort_values(by=['system_linenr', 'departure_time']).drop_duplicates(
        subset=['trip_number'], keep='first')
    dep_time_dict = {k: v for k, v in zip(
        dep_time['trip_number'], dep_time['departure_time'])}

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
        by=['system_linenr', 'departure_time']).drop_duplicates(subset=['trip_number'], keep='first')

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
    model.addConstrs((reassign_var.sum('*', j) <=
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

    for i, j in paired_trips:
        # max number of imposed cancellation
        model.addConstrs((imposed_var.sum(i, '*', k) <=
                         2 for k in trip_following_trips[i]))
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
    return model, active_arcs, imposed_arc


model, active_arcs, imposed_arc = bus_reassginment('2022-09-20')


"""
plot occupancy data
"""

# # print optimal solutions


def conv_millsecond_localtime(milliseconds):
    uct_time = dt.datetime.fromtimestamp(milliseconds/1000.0)
    uct_time = pytz.utc.localize(uct_time)
    cet = pytz.timezone('CET')
    offset = uct_time.astimezone(cet).utcoffset()
    local_time = uct_time - offset
    return local_time


def plot_occupancy(cancelled_trip, reassigned_trip):
    # data['DepartureTime'] = data['DepartureTime'].apply(
    #     conv_millsecond_localtime)
    line_number = data.sort_values(by=['Systeemlijnnr']).drop_duplicates(
        subset=['TripNumber'], keep='first')
    line_number = {i: j for i, j in zip(
        line_number.TripNumber, line_number.Systeemlijnnr)}
    cancelled = data[data['TripNumber'] == cancelled_trip]
    reassigned = data[data['TripNumber'] == reassigned_trip]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(cancelled.ActualDepartureTime,
             cancelled.occupancy, color='green')
    ax1.legend(['Cancelled Trip'], loc=1)
    ax1.set(xlabel='Departure Time', ylabel='Occupancy')
    ax1.title.set_text('Line number: {} \n Trip number: {}'.format(
        line_number[cancelled_trip], cancelled_trip))
    ax1.tick_params(axis="x", rotation=45)

    ax2.plot(reassigned.ActualDepartureTime,
             reassigned.occupancy, color='blue')
    ax2.legend(['Overcrowded Trip'], loc=1)
    ax2.set(xlabel='Departure Time', ylabel='Occupancy')
    ax2.title.set_text('Line number: {} \n Trip number: {}'.format(
        line_number[reassigned_trip], reassigned_trip))
    ax2.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    plt.savefig('Occupancy for trip {} assigned to {}.png'.format(
        cancelled_trip, reassigned_trip), bbox_inches='tight', dpi=300)
    plt.show()


for i, j in active_arcs:
    plot_occupancy(i, j)


# detailes of cancelled and re-assigned trips
def model_output(active_arcs, imposed_arc, data):
    if len(active_arcs) > 0:
        pass
    else:
        sys.exit()
    epsilon = 120000
    cursor, conn = dp.connect_to_database()
    bus_stops = pd.read_sql_query('select * from bus_stops', conn)
    bus_stops = {i: j for i, j in zip(
        bus_stops['stop'], bus_stops['Naam_halte'])}
    cursor.close()

    deadhead_dict = deadhead()
    demand_dict, stranded_passenger_dict, stops_dict, toAssign, reAssign, trip_following_trips, preceeding_trip = sets_parameters(
        data)

    first_stop_dict, last_stop_dict, dep_time_dict, arr_time_dict, travel_time_dict = dp.data_preprocessing(
        data)
    # data['DepartureTime'] = data['DepartureTime'].apply(
    #     conv_millsecond_localtime)

    line_number = data.sort_values(by=['Systeemlijnnr']).drop_duplicates(
        subset=['TripNumber'], keep='first')
    line_number = {i: j for i, j in zip(
        line_number.TripNumber, line_number.Systeemlijnnr)}

    for i, j in active_arcs:
        cancel_linenr = line_number[i]
        cancel_departure = conv_millsecond_localtime(dep_time_dict[40869])
        cancel_first_stop = bus_stops[first_stop_dict[i]]
        reassign_linenr = line_number[j]
        toassign_departure = conv_millsecond_localtime(dep_time_dict[j])
        deadhead_time = (
            deadhead_dict[(first_stop_dict[i], first_stop_dict[j])])/60000
        reassigned_departure = conv_millsecond_localtime(
            dep_time_dict[i] + deadhead_dict[(first_stop_dict[i], first_stop_dict[j])] + epsilon)
        reassigned_first_stop = bus_stops[first_stop_dict[j]]

        for p, q, r in imposed_arc:
            if len(imposed_arc) > 0 and p == i and q == j:
                next_planned_trip = trip_following_trips[r][0]
                return_time = conv_millsecond_localtime(dep_time_dict[p] + deadhead_dict[(first_stop_dict[p], first_stop_dict[q])] +
                                                        epsilon + travel_time_dict[q] + deadhead_dict[(last_stop_dict[q], first_stop_dict[next_planned_trip])])
                ext_planned_linenr = line_number[next_planned_trip]
                next_planned_trip_departure = conv_millsecond_localtime(
                    dep_time_dict[next_planned_trip])
                next_trip_first_stop = bus_stops[first_stop_dict[r]]
            else:
                sys.exit()
        next_planned_trip = trip_following_trips[i][0]
        return_time = conv_millsecond_localtime(dep_time_dict[i] + deadhead_dict[(first_stop_dict[i], first_stop_dict[j])] +
                                                epsilon + travel_time_dict[j] + deadhead_dict[(last_stop_dict[j], first_stop_dict[next_planned_trip])])
        next_planned_trip_departure = conv_millsecond_localtime(
            dep_time_dict[next_planned_trip])
        next_planned_linenr = line_number[next_planned_trip]
        next_trip_first_stop = bus_stops[first_stop_dict[next_planned_trip]]

        print(colored('Optimal solution:', 'green', attrs=['bold']))
        print("Cancelled trip {}".format(i) + ' ' +
              'from line {}'.format(cancel_linenr))
        print("Assign before/after trip {}".format(j) +
              ' ' + 'on line {}'.format(reassign_linenr))
        print(colored('Cancellation details:', 'green', attrs=['bold']))
        print("Planned departure from: {}".format(
            cancel_first_stop) + ': {}'.format(cancel_departure))
        print("Deadhead time from: {}".format(cancel_first_stop) +
              " to: {}".format(reassigned_first_stop) + ": {}".format(deadhead_time) + ' minutes')
        print('Boarding time: 2 minutes')
        print("Re-assigned departure from {}".format(reassigned_first_stop) +
              ': {}'.format(reassigned_departure))
        print("Return to the next planned trip {}".format(
            next_planned_trip) + " " + "on line {}".format(next_planned_linenr) + ': {}'.format(return_time))
        print("Planned departure of {}".format(next_planned_trip) + ' ' +
              'from {}'.format(next_trip_first_stop) + ': {}'.format(next_planned_trip_departure))

        cursor, conn = dp.connect_to_database()
        model_output = pd.read_sql_query('select * from model_output', conn)
        if i not in list(model_output['cancelledTrip']):
            cursor.execute("INSERT INTO model_output (cancelledTrip, lineNumber, plannedDeparture, plannedStop, reassignTo, reassignDeparture, reassignStop, deadheadTime, boardingtime, reassignedDeparture, returnTime, nextTrip, nextLineNumber, nextDeparture, nextStop) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           i, cancel_linenr, cancel_departure, cancel_first_stop, j, toassign_departure, reassigned_first_stop, deadhead_time, epsilon/60000, reassigned_departure, return_time, next_planned_trip, next_planned_linenr, next_planned_trip_departure, next_trip_first_stop)

            conn.commit()
        # cursor.execute("SELECT * FROM model_output GROUP BY cancelledTrip HAVING COUNT(cancelledTrip) > 1")
        cursor.close()


model_output(active_arcs, imposed_arc, data)


# # list of cancelled trips
# cancelled = [a[0] for a in active_arcs]
# reassigned_before = [a[1] for a in active_arcs]


# stranded_pas = quicksum(demand_dict[i, s]
#                         for i in cancelled for s in stops_dict[i])

# model.computeIIS()
# model.write("model.ilp")
