# import libraries
from re import L
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
from datetime import timedelta
from itertools import combinations, product
import statistics
from matplotlib import animation
from IPython.display import HTML
# import dataset
data = pd.read_csv(
    r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/BezettingFare.csv', sep=';')
# data.drop('Unnamed: 0', axis=1, inplace=True)
# replace the NaN values with 0
data['Bezetting'] = data['Bezetting'].fillna(0)

# departure and passing time to datetime format
data['date'] = pd.to_datetime(data['IdDimDatum'].astype(str), format='%Y%m%d')
data['date'] = data.apply(
    lambda row: row['date'] +
    timedelta(1) if row['IdDimTijdBlok'] >= 24 else row['date'],
    axis=1)
data['date'] = pd.to_datetime(data['date']).dt.date

data['passeer_datetime'] = data['date'].map(
    str) + ' ' + data['Passeertijd'].map(str)
data['passeer_datetime'] = pd.to_datetime(
    data['passeer_datetime'], infer_datetime_format=True)

data['dep_datetime'] = data['date'].map(
    str) + ' ' + data['RitVertrekTijd'].map(str)
data['dep_datetime'] = pd.to_datetime(
    data['dep_datetime'], format='%Y-%m-%d %H:%M:%S')


# fig = plt.figure(figsize=(12, 10))
# plt.scatter(data[data['Richting'] == 1]['PublieksLijnnr'],
#             data[data['Richting'] == 1]['dep_datetime'], c='b')
# plt.xlabel('Bus Line')
# plt.ylabel('Departure Time')
# plt.show()


# %%

# calculate in-vehicle crowd exceeding the capacity threshold
capacity_threshold = 35
data["ex_capacity"] = data['Bezetting'].apply(
    lambda x: x - capacity_threshold if (x > capacity_threshold) else 0)

# %%

''' Testing the model only for one day '''

# select the date
test_date = {'2022-02-11'}
test_date = pd.DataFrame(test_date, columns=['date'])

# select the test data set
test_data = data[data['IdDimDatum'] == 20220211]

''' drop line 8 from the test dataset '''
test_data = test_data[test_data['PublieksLijnnr'] != 8]
# test_data['passeer_datetime'] = test_data['passeer_datetime'].apply(conv_time_to_mils)
# test_data['dep_datetime'] = test_data['dep_datetime'].apply(conv_time_to_mils)

# time window-frame of the optimization
# minimum time frame should be one hour and maximum should be one day
# lower bound

min_time = {'07:20:00'}
min_time = pd.DataFrame(min_time, columns=['min_time'])
test_date['min_time'] = min_time['min_time']
test_date['min_datetime'] = test_date['date'].map(
    str) + ' ' + min_time['min_time'].map(str)
test_date['min_datetime'] = pd.to_datetime(
    test_date['min_datetime'], infer_datetime_format=True)
sigma_min = []
sigma_min = test_date['min_datetime']

# upper bound
max_time = {'8:20:00'}
max_time = pd.DataFrame(max_time, columns=['max_time'])
test_date['max_time'] = max_time['max_time']
test_date['max_datetime'] = test_date['date'].map(
    str) + ' ' + max_time['max_time'].map(str)
sigma_max = []
sigma_max = test_date['max_datetime']
# trips in between 5:00 and 23:00
test_data = test_data[test_data.loc[:, 'dep_datetime'] >= sigma_min[0]]
test_data = test_data[test_data.loc[:, 'dep_datetime'] <= sigma_max[0]]


''' Calculating waiting time
parameters:
1. headway mean
2. headway variance '''
sorted_data = test_data.sort_values(by=['PublieksLijnnr', 'dep_datetime']).drop_duplicates(
    subset=['Ritnummer'], keep='first')

sorted_data['h_headway'] = sorted_data.groupby(
    'PublieksLijnnr')['dep_datetime'].transform(pd.Series.diff)
sorted_data['h_headway'] = (sorted_data['h_headway'].dt.total_seconds())/60
sorted_data['h_headway'] = sorted_data['h_headway'].fillna(method='bfill')

sorted_data['h_var'] = sorted_data.groupby(
    'PublieksLijnnr')['h_headway'].transform(statistics.variance)
sorted_data['h_mean'] = sorted_data.groupby(
    'PublieksLijnnr')['h_headway'].transform(statistics.mean)


def cal_waiting_time(mean, var):
    waiting_time = mean * 0.5 + 0.5 * (var / mean)
    return waiting_time


sorted_data['waiting_time'] = cal_waiting_time(
    sorted_data['h_mean'], sorted_data['h_var'])
# waiting time as dictionary
waiting_time_dict = sorted_data.set_index(
    ['Ritnummer']).to_dict()['waiting_time']

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
    return travel_time.dt.total_seconds()/60


arr_time = test_data.loc[test_data.groupby(
    'Ritnummer')['passeer_datetime'].idxmax()]
arr_time['travel_time'] = cal_travel_time(
    arr_time['passeer_datetime'], arr_time['dep_datetime'])
arr_time['passeer_datetime'] = arr_time['passeer_datetime'].apply(
    conv_time_to_mils)
arr_time_dict = arr_time.set_index(['Ritnummer']).to_dict()['passeer_datetime']
travel_time_dict = arr_time.set_index(['Ritnummer']).to_dict()['travel_time']

demand_dict = test_data.sort_values(by=['Ritnummer', 'IdDimHalte']).set_index(
    ['Ritnummer', 'IdDimHalte']).to_dict()['Bezetting']

ex_capacity_dict = test_data.sort_values(by=['Ritnummer', 'IdDimHalte']).set_index(
    ['Ritnummer', 'IdDimHalte']).to_dict()['ex_capacity']
# list of trips
trips = test_data['Ritnummer']
trips = trips.drop_duplicates(keep='first').tolist()

# divide the set of of trips into two subsets A and R
ex_capacity_trip = list(test_data[test_data['ex_capacity'] > 0]['Ritnummer'])

# list of trips exceeding the capacity threshold
toAssign = test_data[test_data['Ritnummer'].isin(ex_capacity_trip)]
toAssign = toAssign['Ritnummer'].drop_duplicates(keep='first').tolist()

# List of trips that could potetially be re-assigned

'''
For creating list B, these preconditions should met:
   1. trips should depart before depature time of trips in A
   2. their following trips should not be overcrowded
'''
reAssign = test_data[~test_data['Ritnummer'].isin(ex_capacity_trip)]
# remove duplicates
reAssign = reAssign['Ritnummer'].drop_duplicates(keep='first').tolist()

stops_dict = {k: list(v)
              for k, v in test_data.groupby('Ritnummer')['IdDimHalte']}
# create the model without function


# def pairs(*lists):
#     for t in combinations(lists, 2):
#         for pair in product(*t):
#             yield pair
# trip_pairs = [pair for pair in pairs(toAssign, reAssign)]

# pairing trips with precondition


def bus_reassginment(toAssign, reAssign, waitingTime, demand, exceedingCapacity, departureTime, arrivalTime, stops):
    model = gp.Model('Bus Reassignment')

    # create pair of potential trips for re-assignemnt and trips that they could be re-assigned before
    paired_trips = tuplelist()
    for i in reAssign:
        for j in toAssign:
            if arrivalTime[i] <= departureTime[j]:
                paired_trips += [(i, j)]
    # x = model.addVars(trip_pairs, vtype=GRB.BINARY, name='x')
    reassign_var = model.addVars(
        paired_trips, vtype=GRB.BINARY, name="x[%s, %s]" % (i, j))
    model.update()
    # add constraints
    model.addConstrs((reassign_var.sum('*', j) <=
                     1 for j in toAssign), name="toAssign[%s]" % j)

    model.addConstrs((reassign_var.sum(i, '*') <=
                     1 for i in reAssign), name='cancellation[%s]' % i)
    model.update()
    # for j in toAssign:
    #     model.addConstr(quicksum(x[i, j]
    #                     for (i, j) in x) == 1, name='toAssign[%s]' % j)
    # for i in reAssign:
    #     model.addConstr(quicksum(x[i, j]
    #                     for (i, j) in x) <= 1, name='reAssgin[%s]' % i)
    # objective
    obj = quicksum(0.5 * reassign_var[i, j] * exceedingCapacity[j, s] * waitingTime[j] for i, j in paired_trips for s in stops[j]) + quicksum(3 * (1-reassign_var[i, j]) * exceedingCapacity[j, s]
                                                                                                                                              * waitingTime[j] for i, j in paired_trips for s in stops[j]) + quicksum(2 * reassign_var[i, j] * demand[i, s] * waitingTime[i] for i, j in paired_trips for s in stops[i])
    model.setObjective(obj, GRB.MINIMIZE)
    model.update()
    model.__data = reassign_var
    return model


model = bus_reassginment(toAssign, reAssign, waiting_time_dict, demand_dict,
                         ex_capacity_dict, dep_time_dict, arr_time_dict, stops_dict)

model.optimize()

# find optimal solutions


def active_arcs(model):
    paired_trips = tuplelist()
    for i in reAssign:
        for j in toAssign:
            if arr_time_dict[i] <= dep_time_dict[j]:
                paired_trips += [(i, j)]
    active_arcs = [a for a in paired_trips if model.__data[a].x > 0.99]
    print("Model's Output: \n")
    for i in active_arcs:
        print(
            "Trip {} can be re-assigned".format(i[0]) + ' ' + "before trip {}".format(i[1]))
    for j in toAssign:
        if j not in [a[1] for a in active_arcs]:
            print("No optimal bus trip was found to re-assign before trip {}".format(j))


active_arcs(model)


# list of cancelled trips
cancelled = [a[0] for a in active_arcs]
reassigned_before = [a[1] for a in active_arcs]


stranded_pas = quicksum(demand_dict[i, s]
                        for i in cancelled for s in stops_dict[i])

# model.computeIIS()
# model.write("model.ilp")
