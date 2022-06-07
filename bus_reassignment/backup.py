

def selected(vals):
    s = {}
    for i, j in vals.keys():
        if vals[i, j] > 0.5:
            s.append((i, j))
    return s


''' Comments:
1. some trips on line 9 (Enschede - Hengelo) do not start from Enschede central station
2. some trips on line 1 (De Posten - UT)
3. Bezetting increases/decreases without changes in the number instappers/outstappers'''

# %%
'''
Model parameters
 '''


# occupancy data
occ_A = A[['Ritnummer', 'IdDimHalte', 'Bezetting']]
occ_A_dict = occ_A.set_index(
    ['Ritnummer', 'IdDimHalte']).to_dict()['Bezetting']

occ_B = B[['Ritnummer', 'IdDimHalte', 'Bezetting']]
occ_B_dict = occ_B.set_index(
    ['Ritnummer', 'IdDimHalte']).to_dict()['Bezetting']

# trip - exceeding capacity threshold
ex_cap_A = A[['Ritnummer', 'IdDimHalte', 'ex_capacity']]
ex_cap_A_dict = ex_cap_A.set_index(
    ['Ritnummer', 'IdDimHalte']).to_dict()['ex_capacity']

ex_cap_B = B[['Ritnummer', 'IdDimHalte', 'ex_capacity']]
ex_cap_B_dict = ex_cap_B.set_index(
    ['Ritnummer', 'IdDimHalte']).to_dict()['ex_capacity']


# calculate deadhead time


# first and last stop for each trip
first_last_stop = test_data[['Ritnummer', 'IdDimHalte',
                             'Naam_halte', 'passeer_datetime', 'dep_datetime']]
first_stop = first_last_stop.loc[first_last_stop.groupby(
    'Ritnummer')['passeer_datetime'].idxmin()]
last_stop = first_last_stop.loc[first_last_stop.groupby(
    'Ritnummer')['passeer_datetime'].idxmax()]


# deadhead cost
# trip arc and travel time between the first and last stop of each line
arc, travel_time = gp.multidict({
    ("ECS", "UT", 1): 35,
    ("UT", "ECS", 1): 32,
    ("ECS", "Wesselerbrink", 1): 25,
    ("Wesselerbrink", "ECS", 1): 32,
    ("ECS", "Deppenbroek", 2): 35,
    ("Deppenbroek", "ECS", 2): 40,
    ("ECS", "Glanerbrug", 3): 50,
    ("Glanerbrug", "ECS", 3): 45,
    ("ECS", "Stroinslanden", 4): 28,
    ("Stroinslanden", "ECS", 4): 33,
    ("ECS", "Zwering", 5): 41,
    ("Zwering", "ECS", 5): 39,
    ("ECS", "Stokhorst", 6): 32,
    ("Stokhorst", "ECS", 6): 28,
    ("ECS", "Marssteden", 7): 42,
    ("Marssteden", "ECS", 7): 48,
    ("ECS", "Hengelo", 9): 35,
    ("Hengelo", "ECS", 9): 35
})


grouped_data = pd.DataFrame(test_data.groupby(['PublieksLijnnr', 'Ritnummer'])[
    'dep_datetime', 'IdDimHalte', 'Naam_halte'].first().reset_index())
grouped_data['dep_datetime'] = grouped_data['dep_datetime'].apply(
    conv_time_to_mils)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")
# ax.scatter(grouped_data['PublieksLijnnr'], grouped_data['IdDimHalte'], grouped_data['dep_datetime'], c='b')
for i in range(len(grouped_data)):
    x = grouped_data.loc[i, 'PublieksLijnnr']
    y = grouped_data.loc[i, 'IdDimHalte']
    z = grouped_data.loc[i, 'dep_datetime']
    label = grouped_data.loc[i, 'Naam_halte']
    ax.scatter(x, y, z, color='b')
    ax.text(x, y, z, '%s' % (label), size=8, color='k')
ax.set_xlabel('Bus line')
ax.set_ylabel('Bus stop')
ax.set_zlabel('Departure time')
plt.show()


# Writing mathematical formulation
"""
Objective 1: minimize waiting time of stranded passengers (re-assignment vs cancellation)

objective 2: minimize deadhead cost (from the last stop of timetabled trip to the first stop of re-assigned trip and back to the first stop of following trip operated by the same bus)

Constratins:
   1. arrival time to the first stop of re-assigned trip <= depature of the crowded trip
   2. Re-assignment: a cancelled trip can be only re-assigned once and only one bus trip is re-assigned before an overcrowded trip
   3. imposed cancellation: If the bus operating the cancelled trip cannot arrive for its next trip, its following trip will also be cancelled.
   4. max imposed cancellation is 2

"""

# %%
# calculate the pairwise potential re-assignment matrix

for t in range(len(dep_time_B)):
    l = [dep_time_B.iloc[t]['Ritnummer'],
         dep_time_B.iloc[t]['passeer_datetime']]
    multi[t] = l
key, trip, dep_time = gp.multidict(multi)


# %%
bus_stops = ['ECS', "UT", "Wesselerbrink", "Deppenbroek", "Glanerbrug", "Stroinslanden",
             "Zwering", "Stokhorst", "Marssteden", "Hengelo"]

# set of trip number
# later this set should be based on the actual data
# each trip has number, first stop, last stop and line number
trip, dep_stop, arr_stop, line_number = gp.multidict({
    1: ["ECS", "UT", 1],
    2: ["UT", "ECS", 1],
    3: ["ECS", "Wesselerbrink", 1],
    4: ["De_Posten", "ECS", 1],
    5: ["ECS", "Deppenbroek", 2],
    6: ["Deppenbroek", "ECS", 2],
    7: ["ECS", "Glanerbrug", 3],
    8: ["Glanerbrug", "ECS", 3],
    9: ["ECS", "Stroinslanden", 4],
    10: ["Stroinslanden", "ECS", 4],
    11: ["ECS", "Zwering", 5],
    12: ["Zwering", "ECS", 5],
    13: ["ECS", "Stokhorst", 6],
    14: ["Stokhorst", "ECS", 6],
    15: ["ECS", "Marssteden", 7],
    16: ["Marssteden", "ECS", 7],
    17: ["ECS", "Hengelo", 9],
    18: ["Hengelo", "ECS", 9]
})

# each trip has departure time from the first stop
# departure time trips
# convert datetime to millisecond


def conv_time_to_mils(time):
    date = datetime.now().date().strftime("%y-%m-%d ")
    date_time = date + time
    date_time = datetime.strptime(
        date_time, "%y-%m-%d %H:%M:%S").timestamp() * 1000
    return date_time


trip, dep_time = gp.multidict({
    1: conv_time_to_mils("09:12:00"),
    2: conv_time_to_mils("09:15:00"),
    3: conv_time_to_mils("09:20:00"),
    4: conv_time_to_mils("09:25:00"),
    5: conv_time_to_mils("09:30:00"),
    6: conv_time_to_mils("09:35:00"),
    7: conv_time_to_mils("09:15:00"),
    8: conv_time_to_mils("09:36:00"),
    9: conv_time_to_mils("09:20:00"),
    10: conv_time_to_mils("09:25:00"),
    11: conv_time_to_mils("09:40:00"),
    12: conv_time_to_mils("09:25:00"),
    13: conv_time_to_mils("09:25:00"),
    14: conv_time_to_mils("09:40:00"),
    15: conv_time_to_mils("09:10:00"),
    16: conv_time_to_mils("09:20:00"),
    17: conv_time_to_mils("09:00:00"),
    18: conv_time_to_mils("09:15:00")
})

# trip arc and travel time between the first and last stop of each line
arc, travel_time = gp.multidict({
    ("ECS", "UT", 1): 35,
    ("UT", "ECS", 1): 32,
    ("ECS", "Wesselerbrink", 1): 25,
    ("Wesselerbrink", "ECS", 1): 32,
    ("ECS", "Deppenbroek", 2): 35,
    ("Deppenbroek", "ECS", 2): 40,
    ("ECS", "Glanerbrug", 3): 50,
    ("Glanerbrug", "ECS", 3): 45,
    ("ECS", "Stroinslanden", 4): 28,
    ("Stroinslanden", "ECS", 4): 33,
    ("ECS", "Zwering", 5): 41,
    ("Zwering", "ECS", 5): 39,
    ("ECS", "Stokhorst", 6): 32,
    ("Stokhorst", "ECS", 6): 28,
    ("ECS", "Marssteden", 7): 42,
    ("Marssteden", "ECS", 7): 48,
    ("ECS", "Hengelo", 9): 35,
    ("Hengelo", "ECS", 9): 35
})
# calculate arrival time


def cal_arr_time(dep_time, travel_time):
    return dep_time + travel_time * 60000


dd = [travel_time[t]
      for t in travel_time.keys() if t == (dep_stop, arr_stop, 1)]

for i in dep_stop.keys():
    key = (dep_stop[i], arr_stop[i], line_number[i])
    if key == ('Zwering', 'ECS', 5):
        print(key)
tt = [t for t in trip]


trip, arr_time = gp.multidict({


    1: cal_arr_time(dep_time[0], [a]),
    2: conv_time_to_mils("09:15:00"),
    3: conv_time_to_mils("09:20:00"),
    4: conv_time_to_mils("09:25:00"),
    5: conv_time_to_mils("09:30:00"),
    6: conv_time_to_mils("09:35:00"),
    7: conv_time_to_mils("09:15:00"),
    8: conv_time_to_mils("09:36:00"),
    9: conv_time_to_mils("09:20:00"),
    10: conv_time_to_mils("09:25:00"),
    11: conv_time_to_mils("09:40:00"),
    12: conv_time_to_mils("09:25:00"),
    13: conv_time_to_mils("09:25:00"),
    14: conv_time_to_mils("09:40:00"),
    15: conv_time_to_mils("09:10:00"),
    16: conv_time_to_mils("09:20:00"),
    17: conv_time_to_mils("09:00:00"),
    18: conv_time_to_mils("09:15:00")
})

# combine two dictionaries
for key, value in dep_stop.items():
    if key in dep_time.keys():
        print(key, value, dep_time[key])

for t in trip:
    return dep_time[t] +


# arc and deadhead time
darc, deadhead_time = gp.multidict({
    ("ECS", "ECS"): 0,
    ("UT", "UT"): 0,
    ("Wesselerbrink", "Wesselerbrink"): 0,
    ("Deppenbroek", "Deppenbroek"): 0,
    ("Glanerbrug", "Glanerbrug"): 0,
    ("Stroinslanden", "Stroinslanden"): 0,
    ("Zwering", "Zwering"): 0,
    ("Stokhorst", "Stokhorst"): 0,
    ("Marssteden", "Marssteden"): 0,
    ("Hengelo", "Hengelo"): 0,
    ("ECS", "UT"): 15,
    ("UT", "ECS"): 15,
    ("ECS", "Wesselerbrink"): 12,
    ("Wesselerbrink", "ECS"): 10,
    ("ECS", "Deppenbroek"): 15,
    ("Deppenbroek", "ECS"): 14,
    ("ECS", "Glanerbrug"): 12,
    ("Glanerbrug", "ECS"): 20,
    ("ECS", "Stroinslanden"): 14,
    ("Stroinslanden", "ECS"): 17,
    ("ECS", "Zwering"): 21,
    ("Zwering", "ECS"): 19,
    ("ECS", "Stokhorst"): 16,
    ("Stokhorst", "ECS"): 14,
    ("ECS", "Marssteden"): 21,
    ("Marssteden", "ECS"): 24,
    ("ECS", "Hengelo"): 17
})

# driving cost per minute
cost = 40  # euros per minute
# Capacity threshold
cap_threshold = 60  # in-vehicle crowd

# compute parameters for the model
num_lines = len(bus_lines)
num_stop = len(bus_stops)
num_trips = len(trips)


# departure time of the first and the last trips
# convert datetime to millisecond
def conv_time_to_mils(time):
    date = datetime.now().date().strftime("%y-%m-%d ")
    date_time = date + time
    date_time = datetime.strptime(
        date_time, "%y-%m-%d %H:%M:%S").timestamp() * 1000
    return date_time


arc, trip, dep_time = gp.multidict({
    ("ECS", "UT"): [1, conv_time_to_mils("09:12:00")],
    ("UT", "ECS"): [2, conv_time_to_mils("09:15:00")],
    ("ECS", "Wesselerbrink"): [3, conv_time_to_mils("09:20:00")],
    ("Wesselerbrink", "ECS"): [4, conv_time_to_mils("09:25:00")],
    ("ECS", "Deppenbroek"): [5, conv_time_to_mils("09:30:00")],
    ("Deppenbroek", "ECS"): [6, conv_time_to_mils("09:35:00")],
    ("ECS", "Glanerbrug"): [7, conv_time_to_mils("09:15:00")],
    ("Glanerbrug", "ECS"): [8, conv_time_to_mils("09:36:00")],
    ("ECS", "Stroinslanden"): [9, conv_time_to_mils("09:20:00")],
    ("Stroinslanden", "ECS"): [10, conv_time_to_mils("09:25:00")],
    ("ECS", "Zwering"): [11, conv_time_to_mils("09:40:00")],
    ("Zwering", "ECS"): [12, conv_time_to_mils("09:25:00")],
    ("ECS", "Stokhorst"): [13, conv_time_to_mils("09:25:00")],
    ("Stokhorst", "ECS"): [14, conv_time_to_mils("09:40:00")],
    ("ECS", "Marssteden"): [15, conv_time_to_mils("09:10:00")],
    ("Marssteden", "ECS"): [16, conv_time_to_mils("09:20:00")],
    ("ECS", "Hengelo"): [17, conv_time_to_mils("09:00:00")]
})


# for each trip, there are three identicators line number, starting stop, and depature time/arrival time
trip = [t for t in trips]
dep_time = {}  # departure time in millisecond
for t in trip:
    for i in bus_lines:
        for j in bus_stops:
            d[t, i, j] = [round(time.time() * 1000)]

a = {}  # arrival time in millisecond
for t in trip:
    for i in L:
        for j in S:
            a[t, i, j] = [round(time.time() * 1000)]

# expected in-vehilce crowding at the most crowded segment
zeta = {}
for t in trip:
    for i in L:
        for j in S:
            zeta[t, i, j] = [random.randint(0, 100)]

# total boarding passengers along all stops within a trip
theta = {}
for t in trip:
    for i in L:
        for j in S:
            theta[t, i, j] = [random.randint(10, 150)]

# maximum capacity threshold
C = 60  # maximum number of in-vehicle passengers


# first of the day
sigma_min = {}
first_trip = datetime.now().date().strftime("%y-%m-%d ") + "05:00:00"
first_trip = datetime.strptime(
    first_trip, "%y-%m-%d %H:%M:%S").timestamp() * 1000

for i in L:
    for j in S:
        sigma_min[i, j] = [first_trip]
# last trip of the day
last_trip = datetime.now().date().strftime("%y-%m-%d ") + "23:00:00"
last_trip = datetime.strptime(
    last_trip, "%y-%m-%d %H:%M:%S").timestamp() * 1000
sigma_max = {}
for i in L:
    for j in S:
        sigma_max[i, j] = [last_trip]

# calculating average waiting time
# w = (E(H)/2)(1+Var(H)/E(H)^2)
# E(H) headway mean
# Var(H) headway variance

m = {}
for i in arc:
    m[i] = [15]


def waiting():
    m = {}
    v = {}
    for i in arc:
        m[i] = [4]
    for i in arc:
        v[i] = [2]


w = waiting()


# model without function
model = gp.Model("Bus Reassignment")
# decision variable
# potential reassignment variable = 1, if i could be re-assigned before j
# create pair of potential trips for re-assignemnt and trips that they could be re-assigned before
paired_trips = tuplelist()
for i in reAssign:
    for j in toAssign:
        if arr_time_dict[i] <= dep_time_dict[j]:
            paired_trips += [(i, j)]

reassign_var = model.addVars(
    paired_trips, vtype=GRB.BINARY, name='reassign_var')
model.update()

# potential imposed cancellation variable = 1 if k is cancelled due to reassignment for j

# there should be at least one trip assigned before overcrowded trip
model.addConstrs((reassign_var.sum('*', j) <=
                 1 for j in toAssign), name='reassignment')
# a trip can only be reassigned once
model.addConstrs((reassign_var.sum(i, '*') <=
                 1 for i in reAssign), name='cancellation')

model.update()

obj = quicksum(0.5 * reassign_var[i, j] * ex_capacity_dict[j, s] * waiting_time_dict[j] for i, j in paired_trips for s in stops_dict[j]) + quicksum(3 * (1-reassign_var[i, j]) * ex_capacity_dict[j, s]
                                                                                                                                                    * waiting_time_dict[j] for i, j in paired_trips for s in stops_dict[j]) + quicksum(2 * reassign_var[i, j] * demand_dict[i, s] * waiting_time_dict[i] for i, j in paired_trips for s in stops_dict[i])

model.setObjective(obj, GRB.MINIMIZE)
model.update()

model.optimize()



# import libraries
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
import googlemaps
gmaps = googlemaps.Client(key='AIzaSyAra0o3L3rs-uHn4EpaXx1Y57SIF_02684')

# import dataset
data = pd.read_csv(
    r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/BezettingFare.csv', sep=';')
# data.drop('Unnamed: 0', axis=1, inplace=True)
# replace the NaN values with 0
data['Bezetting'] = data['Bezetting'].fillna(0)

''' Data Pre processing '''
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

# extract bus trips that depart, pass or arrive to Enschede Central Station
lines_connected_2enschede = [1, 2, 3, 4, 5, 6, 7, 8, 9, 60, 61, 62, 506]
data = data[data['PublieksLijnnr'].isin(lines_connected_2enschede)]


# to find the deadhead time between the last stop a trip and first stop of another trip,
first_stop = data.sort_values(by=['PublieksLijnnr', 'dep_datetime']).drop_duplicates(
    subset=['Ritnummer'], keep='first')
first_stop = first_stop.groupby(['IdDimHalte']).nth(0).reset_index()
first_stop = first_stop[['Ritnummer', 'IdDimHalte', 'Breedtegraad', 'Lengtegraad']]

last_stop = data.sort_values(by=['PublieksLijnnr', 'passeer_datetime']).drop_duplicates(
    subset=['Ritnummer'], keep='last')
last_stop = last_stop.groupby(
    ['PublieksLijnnr', 'Naam_halte']).nth(0).reset_index()
last_stop = last_stop[['Ritnummer', 'IdDimHalte', 'Breedtegraad', 'Lengtegraad']]

deadhead_data = []
for first in first_stop['IdDimHalte']:
    for last in last_stop['IdDimHalte']:
        deadhead_data += [[last, first]]

deadhead_data = pd.DataFrame(deadhead_data, columns=[
                             'last_stop', 'first_stop'])
deadhead_data = deadhead_data.merge(
    first_stop, left_on='last_stop', right_on='IdDimHalte')
deadhead_data.rename(
    columns={'Breedtegraad': 'lat_last', 'Lengtegraad': 'long_last'}, inplace=True)

deadhead_data = deadhead_data.merge(
    last_stop, left_on='first_stop', right_on='IdDimHalte')
deadhead_data.rename(
    columns={'Breedtegraad': 'lat_first', 'Lengtegraad': 'long_first'}, inplace=True)


def calculate_distance(lat1, lon1, lat2, lon2):

    distance = gmaps.distance_matrix([str(lat1) + " " + str(lon1)],
                                     [str(lat2) + " " + str(lon2)],
                                     mode='driving')['rows'][0]['elements'][0]['duration']['text'].split(' ')[0]
    return distance


def f(x): return calculate_distance(
    x['lat_last'], x['long_last'], x['lat_first'], x['long_first'])


deadhead_data['deadhead_time'] = deadhead_data.apply(f, axis=1).astype('float')
# if two stops are located at the same station, the deadhead time is set to zero
deadhead_data['deadhead_time'] = deadhead_data['deadhead_time'].apply(
    lambda x: x * 60000 if x > 1 else 0)

deadhead_data.drop(['IdDimHalte_x', 'IdDimHalte_y'], axis=1, inplace=True)



# %%

''' Testing the model only for one day '''

# select the date
test_date = {'2022-02-11'}
test_date = pd.DataFrame(test_date, columns=['date'])


select_month = 2
select_day = 11
data['month'] = pd.to_datetime(data['date']).dt.month
data['day'] = pd.to_datetime(data['date']).dt.day

# select the test data set
test_data = data[(data['month'] == select_month) & (data['day'] == select_day)]

#%%
''' data pre-processing '''


# preprocessing_data = test_data.sort_values(
#     by=['PublieksLijnnr', 'dep_datetime']).drop_duplicates(subset=['Ritnummer'], keep='first')

# lines_dict = {k: list(v)
#               for k, v in preprocessing_data.groupby('PublieksLijnnr')['Ritnummer']}

# bus_trips_dict = {k: list(v)
#                   for k, v in preprocessing_data.groupby('IdDimVoertuig')['Ritnummer']}


# first_stop_dict = {k: v for k, v in zip(
#     preprocessing_data['Ritnummer'], preprocessing_data['IdDimHalte'])}

# last_stop = test_data.sort_values(by=['PublieksLijnnr', 'dep_datetime']).drop_duplicates(
#     subset=['Ritnummer'], keep='last')

# last_stop_dict = {k: v for k, v in zip(
#     last_stop['Ritnummer'], last_stop['IdDimHalte'])}


# def get_key_from_value(d, val):
#     keys = [k for k, v in d.items() if val in v]
#     if keys:
#         return keys[0]
#     return None

# key = get_key_from_value(bus_trips_dict, 42657)
# print(key)

# test_data['passeer_datetime'] = test_data['passeer_datetime'].apply(conv_time_to_mils)
# test_data['dep_datetime'] = test_data['dep_datetime'].apply(conv_time_to_mils)


# time window-frame of the optimization
# minimum time frame should be one hour and maximum should be one day
# lower bound


min_time = {'07:10:00'}
min_time = pd.DataFrame(min_time, columns=['min_time'])
test_date['min_time'] = min_time['min_time']
test_date['min_datetime'] = test_date['date'].map(
    str) + ' ' + min_time['min_time'].map(str)
test_date['min_datetime'] = pd.to_datetime(
    test_date['min_datetime'], infer_datetime_format=True)
sigma_min = []
sigma_min = test_date['min_datetime']

# upper bound
max_time = {'9:00:00'}
max_time = pd.DataFrame(max_time, columns=['max_time'])
test_date['max_time'] = max_time['max_time']
test_date['max_datetime'] = test_date['date'].map(
    str) + ' ' + max_time['max_time'].map(str)
sigma_max = []
sigma_max = test_date['max_datetime']
# trips in between 5:00 and 23:00
test_data = test_data[test_data.loc[:, 'dep_datetime'] >= sigma_min[0]]
test_data = test_data[test_data.loc[:, 'dep_datetime'] <= sigma_max[0]]


def preprocessing(data):
    line_trips = {}
    bus_trips = {}
    first_stop = {}
    last_stop = {}
    preprocessing_data = data.sort_values(
        by=['PublieksLijnnr', 'dep_datetime']).drop_duplicates(subset=['Ritnummer'], keep='first')
    line_trips = {k: list(v)
                  for k, v in preprocessing_data.groupby('PublieksLijnnr')['Ritnummer']}
    bus_trips = {k: list(v)
                 for k, v in preprocessing_data.groupby('IdDimVoertuig')['Ritnummer']}
    first_stop = {k: v for k, v in zip(
        preprocessing_data['Ritnummer'], preprocessing_data['IdDimHalte'])}
    last_stop = data.sort_values(by=['PublieksLijnnr', 'dep_datetime']).drop_duplicates(
        subset=['Ritnummer'], keep='last')

    last_stop = {k: v for k, v in zip(
        last_stop['Ritnummer'], last_stop['IdDimHalte'])}

    return line_trips, bus_trips, first_stop, last_stop

line_trips_dict, bus_trips_dict, first_stops_dict, last_stops_dict = preprocessing(test_data)

#%%
''' Calculating waiting time
parameters:
1. headway mean
2. headway variance '''
sorted_data = test_data.sort_values(by=['PublieksLijnnr', 'dep_datetime']).drop_duplicates(
    subset=['Ritnummer'], keep='first')
sorted_data['h_headway'] = sorted_data.groupby(by=['PublieksLijnnr', 'IdDimHalte'])[
    'dep_datetime'].transform(pd.Series.diff)

sorted_data['h_headway'] = (sorted_data['h_headway'].dt.total_seconds())/60
sorted_data = sorted_data.sort_values(
    by=['PublieksLijnnr', 'IdDimHalte', 'dep_datetime'])
sorted_data['h_headway'].fillna(method='bfill', inplace=True)

# since variance requires at least two values, remove lines with only on values
sorted_data.groupby(sorted_data.PublieksLijnnr.tolist(), as_index=False).size()

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

#%%

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

#%%

''' list of to assign and re-assign should only from Enschede bus lines '''

# select enschede data
enschede_lines = [1, 2, 3, 4, 5, 6, 7, 9]
enschede_data = test_data[test_data['PublieksLijnnr'].isin(enschede_lines)]
False_data = [41296]
enschede_data = enschede_data[~enschede_data['Ritnummer'].isin(False_data)]
# calculate in-vehicle crowd exceeding the capacity threshold
capacity_threshold = 35
enschede_data["ex_capacity"] = enschede_data['Bezetting'].apply(
    lambda x: x - capacity_threshold if (x > capacity_threshold) else 0)


demand_dict = enschede_data.sort_values(by=['Ritnummer', 'IdDimHalte']).set_index(
    ['Ritnummer', 'IdDimHalte']).to_dict()['Bezetting']

ex_capacity_dict = enschede_data.sort_values(by=['Ritnummer', 'IdDimHalte']).set_index(
    ['Ritnummer', 'IdDimHalte']).to_dict()['ex_capacity']
# # list of trips
# trips = test_data['Ritnummer']
# trips = trips.drop_duplicates(keep='first').tolist()


# divide the set of of trips into two subsets A and R
ex_capacity_trip = list(enschede_data[enschede_data['ex_capacity'] > 0]['Ritnummer'])

# list of trips exceeding the capacity threshold
toAssign = enschede_data[enschede_data['Ritnummer'].isin(ex_capacity_trip)]
toAssign = toAssign['Ritnummer'].drop_duplicates(keep='first').tolist()

# List of trips that could potetially be re-assigned

'''
For creating list B, these preconditions should met:
   1. trips should depart before depature time of trips in A
   2. their following trips should not be overcrowded
'''
reAssign = enschede_data[~enschede_data['Ritnummer'].isin(ex_capacity_trip)]
# remove duplicates
reAssign = reAssign['Ritnummer'].drop_duplicates(keep='first').tolist()

# list of all stops 
stops_dict = {i:k for i, k in zip(test_data['Ritnummer'], test_data['IdDimHalte'])}
# deadhead time dictionary: from stop a to stop b 
deadhead_dict = {(i,j): k for i, j, k in zip(deadhead_data.last_stop, deadhead_data.first_stop, deadhead_data.deadhead_time)}
# # first stop dictionary 
# first_stop_dict = {i:k for i,k in zip(first_stop.Ritnummer, first_stop.IdDimHalte)}
# # last stop dictionary 
# last_stop_dict = {i:k for i,k in zip(last_stop.Ritnummer, last_stop.IdDimHalte)}

# list of following trips for each reAssign trips
paired_trips = tuplelist()
for i in reAssign:
    for j in toAssign:
        stop1 = last_stops_dict[i]
        stop2 = first_stops_dict[j]
        if arr_time_dict[i] +  deadhead_dict[(stop1, stop2)]  <= dep_time_dict[j]:
            paired_trips += [(i, j)]



# def pairs(*lists):
#     for t in combinations(lists, 2):
#         for pair in product(*t):
#             yield pair
# trip_pairs = [pair for pair in pairs(toAssign, reAssign)]

# pairing trips with precondition


def bus_reassginment(toAssign, reAssign, waitingTime, demand, exceedingCapacity, stops):
    model = gp.Model('Bus Reassignment')
    epsilon = 120000  # this is the time for boarding passengers
    # create pair of potential trips for re-assignemnt and trips that they could be re-assigned before
    paired_trips = tuplelist()
    for i in reAssign:
        for j in toAssign:
            stop1 = last_stops_dict[i]
            stop2 = first_stops_dict[j]
            if arr_time_dict[i] +  deadhead_dict[(stop1, stop2)]  + epsilon <= dep_time_dict[j]:
                paired_trips += [(i, j)]
    # x = model.addVars(trip_pairs, vtype=GRB.BINARY, name='x')
    reassign_var = model.addVars(
        paired_trips, vtype=GRB.BINARY, name="x[%s, %s]" % (i, j))
    # create pair of potential imposed cancellations
    imposed_paired = tuplelist()

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
    obj = quicksum(0.5 * reassign_var[i, j] * exceedingCapacity[j, s] * waitingTime[j] for i, j in paired_trips for s in stops[j]) + quicksum(3 * (1-reassign_var[i, j]) * exceedingCapacity[j, s]* waitingTime[j] for i, j in paired_trips for s in stops[j]) + quicksum(2 * reassign_var[i, j] * demand[i, s] * waitingTime[i] for i, j in paired_trips for s in stops[i])
    model.setObjective(obj, GRB.MINIMIZE)
    model.update()
    model.__data = reassign_var
    return model


model = bus_reassginment(toAssign, reAssign, waiting_time_dict, demand_dict,
                         ex_capacity_dict, stops_dict)

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
mos

stranded_pas = quicksum(demand_dict[i, s]
                        for i in cancelled for s in stops_dict[i])

# model.computeIIS()
# model.write("model.ilp")

''' Plot feasible connection of trips with respect to depature time and XY locations '''




model = gp.Model("Bus Reassignment")
# decision variable
# potential reassignment variable = 1, if i could be re-assigned before j
# create pair of potential trips for re-assignemnt and trips that they could be re-assigned before
paired_trips = tuplelist()
for i in reAssign:
    for j in toAssign:
        stop1 = last_stops_dict[i]
        stop2 = first_stops_dict[j]
        if arr_time_dict[i] +  deadhead_dict[(stop1, stop2)] <= dep_time_dict[j]:
            paired_trips += [(i, j)]


reassign_var = model.addVars(
    paired_trips, vtype=GRB.BINARY, name='reassign_var')
model.update()

# potential imposed cancellation variable = 1 if k is cancelled due to reassignment for j

# there should be at least one trip assigned before overcrowded trip
model.addConstrs((reassign_var.sum('*', j) <=
                 1 for j in toAssign), name='reassignment')
# a trip can only be reassigned once
model.addConstrs((reassign_var.sum(i, '*') <=
                 1 for i in reAssign), name='cancellation')

model.update()
test = quicksum(ex_capacity_dict[(i,s)] for i in toAssign for s in stops_dict[i])

obj = quicksum(0.5 * reassign_var[(i, j)] * ex_capacity_dict[(j, s)] * waiting_time_dict[j] for i, j in paired_trips for s in stops_dict[j])

 + quicksum(3 * (1-reassign_var[i, j]) * ex_capacity_dict[j, s] * waiting_time_dict[j] for i, j in paired_trips for s in stops_dict[j]) + quicksum(2 * reassign_var[i, j] * demand_dict[i, s] * waiting_time_dict[i] for i, j in paired_trips for s in stops_dict[i])

model.setObjective(obj, GRB.MINIMIZE)
model.update()

model.optimize()

stops_dict = {k: list(v)
              for k, v in test_data.groupby('Ritnummer')['IdDimHalte']}