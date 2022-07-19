# import libraries
from soupsieve import select
import data_preprocessing as dp
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
gmaps = googlemaps.Client(key='AIzaSyAra0o3L3rs-uHn4EpaXx1Y57SIF_02684')

# AIzaSyBmoRMRmfpYO5AG3GsSbmKlyNPaWOkKROE
# import functions
# insert year, month, and day


def deadhead():
    cursor, conn = dp.connect_to_database()
    deadhead_time = pd.read_sql_query('select * from deadhead_time', conn)
    deadhead_dict = {}

    deadhead_dict.update({(i, j): k for i, j, k in zip(
        deadhead_time.stopA, deadhead_time.stopB, deadhead_time.deadhead)})

    return deadhead_dict

# %%


def conv_time_to_mils(date_time):
    return date_time.timestamp() * 1000

# %%


''' list of to assign and re-assign should only from Enschede bus lines '''


def sets_parameters(data):
    capacity_threshold = 35
    demand_dict = {}  # number of expected boarding passengers at every stop
    # number of passengers who cannot board buses due to overcrowding
    stranded_passenger_dict = {}
    stops_dict = {}  # list of all stops along each trip
    trip_following_trips = {}
    bus_trips = {}  # list trips operated by the same bus
    line_trips = {}  # list of trips on the same bus line
    # convert timestamp into milliseconds
    # data['DepartureTime'] = data['DepartureTime'].apply(
    #     conv_time_to_mils)

    # remove the first and last trip of the day
    first_trip = data.sort_values(by=['Systeemlijnnr', 'DepartureTime']).drop_duplicates(
        subset=['TripNumber'], keep='first')
    first_trip = first_trip.loc[first_trip.groupby(
        by=['Systeemlijnnr', 'Direction']).DepartureTime.idxmin()]
    first_trip = first_trip['TripNumber'].tolist()
    last_trip = data.sort_values(by=['Systeemlijnnr', 'DepartureTime']).drop_duplicates(
        subset=['TripNumber'], keep='last')
    last_trip = last_trip.loc[last_trip.groupby(
        by=['Systeemlijnnr', 'Direction']).DepartureTime.idxmax()]
    last_trip = last_trip['TripNumber'].tolist()
    data = data[~data['TripNumber'].isin(first_trip)]
    data = data[~data['TripNumber'].isin(last_trip)]

    # calculate occupancy exceeding the capacity threshold
    data['exceeding_capacity'] = data['OccupancyCorrected'].apply(
        lambda x: x - capacity_threshold if (x > capacity_threshold) else 0)

    demand_dict = {(i, j): k for i, j, k in zip(
        data['TripNumber'], data['IdDimHalte'], data['OccupancyCorrected'])}

    stranded_passenger_dict = data.sort_values(by=['TripNumber', 'IdDimHalte']).set_index(
        ['TripNumber', 'IdDimHalte']).to_dict()['exceeding_capacity']

    stops_dict = {k: list(v)
                  for k, v in data.groupby('TripNumber')['IdDimHalte']}

    enschede_lines = [4701, 4702, 4703, 4704, 4705, 4706, 4707, 4709]
    enschede = data[data['Systeemlijnnr'].isin(enschede_lines)]
    # list of trips exceeding the capacity threshold
    trips_ex_capacity = [
        k for k in enschede[enschede['exceeding_capacity'] > 0]['TripNumber']]
    toAssign = enschede[enschede['TripNumber'].isin(trips_ex_capacity)]
    toAssign = toAssign['TripNumber'].drop_duplicates(keep='first').tolist()

    # list of trips that could potentially be re-assigned
    reAssign = enschede[~enschede['TripNumber'].isin(trips_ex_capacity)]
    reAssign = reAssign['TripNumber'].drop_duplicates(keep='first').tolist()

    # list of three following trips for each trip operated by the same bus
    sorted_data = data.sort_values(
        by=['IdVehicle', 'DepartureTime']).drop_duplicates(subset=['TripNumber'], keep='first')

    bus_trips.update({k: list(v)
                      for k, v in sorted_data.groupby('IdVehicle')['TripNumber']})

    all_following_trips = {}
    for trip in reAssign:
        list_ftrips = []
        for key, value in bus_trips.items():
            if trip in value:
                all_following_trips.update({trip: value})

    dep_time = data.sort_values(by=['Systeemlijnnr', 'DepartureTime']).drop_duplicates(
        subset=['TripNumber'], keep='first')
    dep_time_dict = {k: v for k, v in zip(
        dep_time['TripNumber'], dep_time['DepartureTime'])}

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
        by=['Systeemlijnnr', 'DepartureTime']).drop_duplicates(subset=['TripNumber'], keep='first')

    line_trips.update({k: list(v)
                       for k, v in sorted_data2.groupby(['Systeemlijnnr', 'Direction'])['TripNumber']})

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

    return demand_dict, stranded_passenger_dict, stops_dict, toAssign, reAssign, trip_following_trips, preceeding_trip


'''
For creating list B, these preconditions should met:
   1. trips should depart before depature time of trips in A
   2. their following trips should not be overcrowded
'''

data = dp.select_data(2022, 2, 12)


def bus_reassginment(data):
    time_window = 60000
    deadhead_dict = deadhead()
    demand_dict, stranded_passenger_dict, stops_dict, toAssign, reAssign, trip_following_trips, preceeding_trip = sets_parameters(
        data)

    waiting_time_dict = dp.waiting_time(data)

    first_stop_dict, last_stop_dict, dep_time_dict, arr_time_dict, travel_time_dict = dp.data_preprocessing(
        data)

    model = gp.Model('Bus Reassignment')
    epsilon = 120000  # this is the time for boarding passengers

    bigM = 1.64456466e+12  # big value
    deadhead_threshold = 900000  # deadhead time threshold is set 15 minutes
    # create pair of potential trips for re-assignemnt and trips that they could be re-assigned before
    if len(toAssign) > 0:
        print('Found overcrowded trips')
    else:
        print(colored("Error! No overcrowded trip was found",
              'green', attrs=['bold']))
        sys.exit()

    paired_trips = tuplelist()
    for i in reAssign:
        for j in toAssign:
            if dep_time_dict[i] + deadhead_dict[(first_stop_dict[i], first_stop_dict[j])] <= dep_time_dict[j] + time_window:
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
    obj = quicksum(0.5 * reassign_var[i, j] * stranded_passenger_dict[j, s] * waiting_time_dict[j] for i, j in paired_trips for s in stops_dict[j]) + quicksum(3 * (1-reassign_var[i, j]) * stranded_passenger_dict[j, s] * waiting_time_dict[j] for i, j in paired_trips for s in stops_dict[j]) + quicksum(2 * reassign_var[i, j] *
                                                                                                                                                                                                                                                                                                             demand_dict[i, s] * waiting_time_dict[i] for i, j in paired_trips for s in stops_dict[i]) + quicksum(2 * imposed_var[i, j, k] * demand_dict[k, s] * waiting_time_dict[k] for i, j in paired_trips for k in trip_following_trips[i] for s in stops_dict[k])  # + quicksum(0.001 * imposed_var[i,j,k] for i, j, k in imposed_cancellation)
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
    model.addConstrs((reassign_var[i, j] * (dep_time_dict[i] + deadhead_dict[(first_stop_dict[i], first_stop_dict[j])])
                     <= dep_time_dict[j] + time_window for i, j in paired_trips), name='departureTimeUp[%s, %s]' % (i, j))
    # model.addConstrs((reassign_var[i, j] * (dep_time_dict[i] + deadhead_dict[(first_stop_dict[i], first_stop_dict[j])])
    #                  >= dep_time_dict[j] - time_window for i, j in paired_trips), name='departureTimeUp[%s, %s]' % (i, j))
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


model, active_arcs, imposed_arc = bus_reassginment(data)


# # print optimal solutions
def con_milsec_datetime(x):
    date_time = dt.datetime.fromtimestamp(x/1000.0)
    return date_time
# detailes of cancelled and re-assigned trips


epsilon = 120000
for a in active_arcs:
    print(colored('Description of cancelled trip(s)', 'green', attrs=['bold']))
    print("Trip number: {}".format(a[0]))
    line_number = trip_line_dict[a[0]]
    dep_time = con_milsec_datetime(dep_time_dict[a[0]])
    deadhead_time = (
        deadhead[(first_stops_dict[a[1]], first_stops_dict[a[0]])])/60000
    reassigned_dep = con_milsec_datetime(
        dep_time_dict[a[0]] + deadhead[(first_stops_dict[a[1]], first_stops_dict[a[0]])] + epsilon)
    next_planned_trip = following_trips_dict[a[0]][0]
    return_to_next_trip = con_milsec_datetime(dep_time_dict[a[0]] + travel_time_dict[a[1]] + deadhead[(
        first_stops_dict[a[1]], first_stops_dict[a[0]])] + epsilon + deadhead[(last_stops_dict[a[1]], first_stops_dict[next_planned_trip])])
    next_planned_trip_dep = con_milsec_datetime(
        dep_time_dict[next_planned_trip])
    print("Bus line number: {}".format(line_number))
    print("Original departure from the first stop: {}".format(dep_time))
    print("Bus deadhead time from the first stop of {}".format(a[0]) +
          " to the first stop of trip {}".format(a[1]) + ' ' + "(min): {}".format(deadhead_time))
    print("Re-assigned departure from the first stop of trip {}".format(
        a[1]) + ' : {}'.format(reassigned_dep))
    print("Return to the first stop of next planned trip {}".format(
        next_planned_trip) + ":" + " {}".format(return_to_next_trip))
    print("Departure time of next planned trip {}".format(
        a[0]) + "" + ": {}".format(next_planned_trip_dep))

# # list of cancelled trips
# cancelled = [a[0] for a in active_arcs]
# reassigned_before = [a[1] for a in active_arcs]


# stranded_pas = quicksum(demand_dict[i, s]
#                         for i in cancelled for s in stops_dict[i])

# model.computeIIS()
# model.write("model.ilp")
