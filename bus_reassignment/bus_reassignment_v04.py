# import libraries
from data_preprocessing import preprocesing, select_data
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
gmaps = googlemaps.Client(key='AIzaSyAra0o3L3rs-uHn4EpaXx1Y57SIF_02684')

# insert year, month, and day
test_data = select_data(2022, 2, 11)

# %%
''' data pre-processing '''


def preprocessing(data):
    line_trips = {}
    bus_trips = {}
    first_stop = {}
    last_stop = {}
    preprocessing_data = data.sort_values(
        by=['Systeemlijnnr', 'dep_datetime']).drop_duplicates(subset=['Ritnummer'], keep='first')
    line_trips = {k: list(v)
                  for k, v in preprocessing_data.groupby('Systeemlijnnr')['Ritnummer']}
    trip_line = {k: v for k, v in zip(
        preprocessing_data['Ritnummer'], preprocessing_data['Systeemlijnnr'])}
    bus_trips = {k: list(v)
                 for k, v in preprocessing_data.groupby('IdDimVoertuig')['Ritnummer']}
    first_stop = {k: v for k, v in zip(
        preprocessing_data['Ritnummer'], preprocessing_data['IdDimHalte'])}
    last_stop = data.sort_values(by=['Systeemlijnnr', 'dep_datetime']).drop_duplicates(
        subset=['Ritnummer'], keep='last')

    last_stop = {k: v for k, v in zip(
        last_stop['Ritnummer'], last_stop['IdDimHalte'])}

    return line_trips, trip_line, bus_trips, first_stop, last_stop


line_trips_dict, trip_line_dict, bus_trips_dict, first_stops_dict, last_stops_dict = preprocessing(
    test_data)

# %%
''' Calculating waiting time
parameters:
1. headway mean
2. headway variance '''
sorted_data = test_data.sort_values(by=['Systeemlijnnr', 'dep_datetime']).drop_duplicates(
    subset=['Ritnummer'], keep='first')
sorted_data['h_headway'] = sorted_data.groupby(by=['Systeemlijnnr', 'IdDimHalte'])[
    'dep_datetime'].transform(pd.Series.diff)

sorted_data['h_headway'] = (sorted_data['h_headway'].dt.total_seconds())/60
sorted_data = sorted_data.sort_values(
    by=['Systeemlijnnr', 'IdDimHalte', 'dep_datetime'])
sorted_data['h_headway'].fillna(method='bfill', inplace=True)

# since variance requires at least two values, remove lines with only on values
sorted_data.groupby(sorted_data.Systeemlijnnr.tolist(), as_index=False).size()

sorted_data['h_var'] = sorted_data.groupby(
    'Systeemlijnnr')['h_headway'].transform(statistics.variance)

sorted_data['h_mean'] = sorted_data.groupby(
    'Systeemlijnnr')['h_headway'].transform(statistics.mean)


def cal_waiting_time(mean, var):
    waiting_time = mean * 0.5 + 0.5 * (var / mean)
    return waiting_time


sorted_data['waiting_time'] = cal_waiting_time(
    sorted_data['h_mean'], sorted_data['h_var'])
# waiting time as dictionary
waiting_time_dict = sorted_data.set_index(
    ['Ritnummer']).to_dict()['waiting_time']

# %%

# depature time of trips from the first stop
# # convert datetime to millisecond


def conv_time_to_mils(date_time):
    return date_time.timestamp() * 1000


sorted_data['dep_datetime'] = sorted_data['dep_datetime'].apply(
    conv_time_to_mils)
dep_time_dict = sorted_data.set_index(['Ritnummer']).to_dict()['dep_datetime']


# arrival to the last stop and travel time
# calculate travel time over each trip
def cal_travel_time(arr_time, dep_time):
    travel_time = ''
    travel_time = arr_time - dep_time
    return travel_time.dt.total_seconds() * 1000


arr_time = test_data.loc[test_data.groupby(
    'Ritnummer')['passeer_datetime'].idxmax()]
arr_time['travel_time'] = cal_travel_time(
    arr_time['passeer_datetime'], arr_time['dep_datetime'])
arr_time['passeer_datetime'] = arr_time['passeer_datetime'].apply(
    conv_time_to_mils)
arr_time_dict = arr_time.set_index(['Ritnummer']).to_dict()['passeer_datetime']
travel_time_dict = arr_time.set_index(['Ritnummer']).to_dict()['travel_time']

# %%

''' list of to assign and re-assign should only from Enschede bus lines '''


# calculate in-vehicle crowd exceeding the capacity threshold
capacity_threshold = 35
test_data["ex_capacity"] = test_data['Bezetting'].apply(
    lambda x: x - capacity_threshold if (x > capacity_threshold) else 0)


demand_dict = test_data.sort_values(by=['Ritnummer', 'IdDimHalte']).set_index(
    ['Ritnummer', 'IdDimHalte']).to_dict()['Bezetting']

# divide the set of of trips into two subsets A and R
ex_capacity_trip = list(
    test_data[test_data['ex_capacity'] > 0]['Ritnummer'])

ex_capacity_dict = test_data.sort_values(by=['Ritnummer', 'IdDimHalte']).set_index(
    ['Ritnummer', 'IdDimHalte']).to_dict()['ex_capacity']

# list of all stops
stops_dict = {k: list(v)
              for k, v in test_data.groupby('Ritnummer')['IdDimHalte']}

# select enschede data
enschede_lines = [1, 2, 3, 4, 5, 6, 7, 9]
enschede_data = test_data[test_data['Systeemlijnnr'].isin(enschede_lines)]


# False_data = [41296]
# enschede_data = enschede_data[~enschede_data['Ritnummer'].isin(False_data)]

# list of overcrowded trips
toAssign = enschede_data[enschede_data['Ritnummer'].isin(ex_capacity_trip)]
toAssign = toAssign['Ritnummer'].drop_duplicates(keep='first').tolist()

# List of trips that could potetially be re-assigned
reAssign = enschede_data[~enschede_data['Ritnummer'].isin(ex_capacity_trip)]

# remove duplicates
reAssign = reAssign['Ritnummer'].drop_duplicates(keep='first').tolist()

# list of following trips for each trip operated by the same bus
trip_bus = test_data.sort_values(
    by=['IdDimVoertuig', 'dep_datetime']).drop_duplicates(subset=['Ritnummer'], keep='first')

trip_bus_dict = {k: v for k, v in zip(
    trip_bus['Ritnummer'], trip_bus['IdDimVoertuig'])}

bus_trip_dict = {k: list(v) for k, v in trip_bus.groupby(
    'IdDimVoertuig')['Ritnummer']}

trip_reAssign = trip_bus[trip_bus['Ritnummer'].isin(reAssign)]

trip_reAssign_dict = {k: v for k, v in zip(
    trip_reAssign['Ritnummer'], trip_reAssign['IdDimVoertuig'])}

trip_ftrips = tuplelist()
for retrip, bus in trip_reAssign_dict.items():
    trip_ftrips += [(retrip, bus_trip_dict[bus])]
trip_ftrips_dict = dict(trip_ftrips)

following_trips = tuplelist()
for key, value in trip_ftrips_dict.items():
    for i in value:
        if dep_time_dict[key] < dep_time_dict[i]:
            following_trips += [(key, i)]

following_trips_dict = {}
for i in following_trips:
    following_trips_dict.setdefault(i[0], []).append(i[1])

# remove trips which doesn't have following trips
reAssign = list(following_trips_dict.keys())
# list of trips that the next trip for that bus is overcrowded
not_reAssign = []
for key, value in following_trips_dict.items():
    for i in toAssign:
        if i in value:
            not_reAssign += [key]

reAssign = [x for x in reAssign if x not in not_reAssign]

# list of the very preceeding trip for each toAssign trip on the same line
line_trip = test_data.sort_values(
    by=['Systeemlijnnr', 'dep_datetime']).drop_duplicates(subset=['Ritnummer'], keep='first')

length = len(line_trip)
preceeding_trip = tuplelist()
for i in range(0, length-1):
    if (line_trip.iloc[i, 4] == line_trip.iloc[i+1, 4]) and (line_trip.iloc[i+1, 6] in toAssign):
        preceeding_trip += [(line_trip.iloc[i+1, 6], line_trip.iloc[i, 6])]
    i+1
preceeding_trip_dict = dict(preceeding_trip)


'''
For creating list B, these preconditions should met:
   1. trips should depart before depature time of trips in A
   2. their following trips should not be overcrowded
'''


def bus_reassginment(toAssign, reAssign, waitingTime, demand, exceedingCapacity, stops):
    model = gp.Model('Bus Reassignment')
    epsilon = 120000  # this is the time for boarding passengers
    bigM = 1.64456466e+13  # big value
    deadhead_threshold = 900000  # deadhead time threshold is set 15 minutes
    # create pair of potential trips for re-assignemnt and trips that they could be re-assigned before
    paired_trips = tuplelist()
    for i in reAssign:
        for j in toAssign:
            stop1 = first_stops_dict[i]
            stop2 = first_stops_dict[j]
            if dep_time_dict[i] + deadhead[(stop1, stop2)] + epsilon <= dep_time_dict[j]:
                paired_trips += [(i, j)]
    # x = model.addVars(trip_pairs, vtype=GRB.BINARY, name='x')
    reassign_var = model.addVars(
        paired_trips, vtype=GRB.BINARY, name="x[%s, %s]" % (i, j))
    # create pair of potential imposed cancellations
    imposed_cancellation = tuplelist()
    for i, j in paired_trips:
        for k in following_trips_dict[i]:
            # if (dep_time_dict[i] + deadhead[(first_stops_dict[i], first_stops_dict[j])] + travel_time_dict[j] + deadhead[(last_stops_dict[j], first_stops_dict[k])] <= dep_time_dict[k]):
            imposed_cancellation += [(i, j, k)]
    imposed_var = model.addVars(
        imposed_cancellation, vtype=GRB.BINARY, name="x[%s, %s, %s]" % (i, j, k))
    model.update()
    # objective
    obj = quicksum(0.5 * reassign_var[i, j] * exceedingCapacity[j, s] * waitingTime[j] for i, j in paired_trips for s in stops[j]) + quicksum(3 * (1-reassign_var[i, j]) * exceedingCapacity[j, s] * waitingTime[j] for i, j in paired_trips for s in stops[j]) + quicksum(2 * reassign_var[i, j] * demand[i, s]
                                                                                                                                                                                                                                                                           * waitingTime[i] for i, j in paired_trips for s in stops[i]) + quicksum(2 * imposed_var[i, j, k] * demand[k, s] * waitingTime[k] for i, j in paired_trips for k in following_trips_dict[i] for s in stops[k])  # + quicksum(0.001 * imposed_var[i,j,k] for i, j, k in imposed_cancellation)
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
    model.addConstrs((reassign_var[i, j] * deadhead[(first_stops_dict[i], first_stops_dict[j])]
                     <= deadhead_threshold for i, j in paired_trips), name='deadhead[%s, %s]' % (i, j))
    # lastest departure time of re-assigned trips
    model.addConstrs((reassign_var[i, j] * (dep_time_dict[i] + deadhead[(first_stops_dict[i], first_stops_dict[j])])
                     <= dep_time_dict[j] for i, j in paired_trips), name='departureTimeUp[%s, %s]' % (i, j))
    # earliest departure time of re-assigned trips: before departure of the very first preceeding trip
    model.addConstrs((reassign_var[i, j] * (dep_time_dict[preceeding_trip_dict[j]]) <= dep_time_dict[i] + deadhead[(first_stops_dict[i], first_stops_dict[j])]
                      for i, j in paired_trips), name='departureTimeUp[%s, %s]' % (i, j))
    model.addConstrs((- reassign_var[i, j] + imposed_var[i, j, k] <=
                     0 for i, j in paired_trips for k in following_trips_dict[i]))
    # imposed cancellation
    for i, j in paired_trips:
        for k in following_trips_dict[i]:
            model.addConstr((reassign_var[i, j] * (dep_time_dict[i] + deadhead[(first_stops_dict[i], first_stops_dict[j])] + travel_time_dict[j] + deadhead[(
                last_stops_dict[j], first_stops_dict[k])])) - bigM * imposed_var[i, j, k] <= dep_time_dict[k])

    for i, j in paired_trips:
        # max number of imposed cancellation
        model.addConstrs((imposed_var.sum(i, '*', k) <=
                         2 for k in following_trips_dict[i]))
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


mode, active_arcs, imposed_arc = bus_reassginment(
    toAssign, reAssign, waiting_time_dict, demand_dict, ex_capacity_dict, stops_dict)


# # print optimal solutions
def con_milsec_datetime(x):
    date_time = dt.datetime.fromtimestamp(x/1000.0)
    return date_time


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
