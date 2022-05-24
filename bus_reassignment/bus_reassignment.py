# import libraries
import gurobipy as gp
from gurobipy import GRB
from gurobipy import multidict, tuplelist, quicksum
import numpy as np
import pandas as pd
from pygments import highlight 
import scipy.sparse as sp
import time
from datetime import datetime
import random
import pyodbc
from itertools import combinations, product
import statistics
# import dataset 
data = pd.read_csv(r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/data_enschede.csv', sep=';')
data.drop('Unnamed: 0', axis=1, inplace=True)
# replace the NaN values with 0
data['Bezetting'] = data['Bezetting'].fillna(0)
# departure and passing time to datetime format 
data['passeer_datetime'] = data['date'].map(str) + ' ' + data['Passeertijd'].map(str) 
data['passeer_datetime'] =  pd.to_datetime(data['passeer_datetime'], infer_datetime_format=True)

data['dep_datetime'] = data['date'].map(str) + ' ' + data['RitVertrekTijd'].map(str) 
data['dep_datetime'] =  pd.to_datetime(data['dep_datetime'], infer_datetime_format=True)

# calculate in-vehicle crowd exceeding the capacity threshold 
capacity_threshold = 50
data["ex_capacity"] = data['Bezetting'].apply(lambda x: x - capacity_threshold if (x > capacity_threshold) else 0)

#%%
''' Testing the model only for one day '''
# select the date
test_date = {'2022-02-11'}
test_date = pd.DataFrame(test_date, columns=['date'])

# select the test data set 
test_data = data[data['IdDimDatum'] == 20220211]

# convert datetime to millisecond 
# def conv_time_to_mils(date_time):
#     return date_time.timestamp() * 1000

# test_data['passeer_datetime'] = test_data['passeer_datetime'].apply(conv_time_to_mils)
# test_data['dep_datetime'] = test_data['dep_datetime'].apply(conv_time_to_mils)

# time window-frame of the optimization
# lower bound
min_time = {'07:30:00'}
min_time = pd.DataFrame(min_time, columns=['min_time'])
test_date['min_time'] = min_time['min_time']
test_date['min_datetime'] = test_date['date'].map(str) + ' ' + min_time['min_time'].map(str) 
test_date['min_datetime'] =  pd.to_datetime(test_date['min_datetime'], infer_datetime_format=True)
sigma_min = []
sigma_min = test_date['min_datetime']


# upper bound
max_time = {'8:30:00'}
max_time = pd.DataFrame(max_time, columns=['max_time'])
test_date['max_time'] = max_time['max_time']
test_date['max_datetime'] = test_date['date'].map(str) + ' ' + max_time['max_time'].map(str) 
sigma_max = []
sigma_max = test_date['max_datetime']
# trips in between 5:00 and 23:00
test_data = test_data[test_data.loc[:, 'dep_datetime'] >= sigma_min[0]]
test_data = test_data[test_data.loc[:, 'dep_datetime'] <= sigma_max[0]]

test_data = test_data.sort_values(by=['Ritnummer', 'dep_datetime'], ascending=[True, False])

# export data for Sander
test_data.to_csv(r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/test_data.csv', sep=";")
#%%
''' Sets and Parameters of the model '''
# list of trips
trip_list = test_data['Ritnummer']
trip_list = trip_list.drop_duplicates(keep='first').tolist()

# extract list of trip numbers where the demand exceeds the capacity threshoold 
ex_capacity_trip = list(test_data[test_data['ex_capacity'] > 0]['Ritnummer'])

# list of trips exceeding the capacity threshold
A = test_data[test_data['Ritnummer'].isin(ex_capacity_trip)]
# remove duplicates and keep the one with the highest values of Bezetting
A = A.sort_values('Bezetting').drop_duplicates(subset=['Naam_halte', 'Passeertijd'], keep='last')
A = A.sort_values(by=['Ritnummer', 'Passeertijd'], ascending=[True, False])

# List of trips that could potetially be re-assigned 

'''
For creating list B, these preconditions should met:
   1. trips should depart before depature time of trips in A
   2. their following trips should not be overcrowded 
'''
B = test_data[~test_data['Ritnummer'].isin(ex_capacity_trip)]
# remove duplicates 
B = B.sort_values('Bezetting').drop_duplicates(subset=['Naam_halte', 'Passeertijd'], keep='last')
B = B.sort_values(by=['Ritnummer', 'Passeertijd'], ascending=[True, False])


''' Comments:
1. some trips on line 9 (Enschede - Hengelo) do not start from Enschede central station
2. some trips on line 1 (De Posten - UT)
3. Bezetting increases/decreases without changes in the number instappers/outstappers'''

#%%
''' 
Model parameters
 '''

# pairing list of trips
trips_A = A['Ritnummer'].drop_duplicates(keep='first').tolist()
trips_B = B['Ritnummer'].drop_duplicates(keep='first').tolist()

def pairs(*lists):
    for t in combinations(lists, 2):
        for pair in product(*t):
            yield pair

trip_pairs = [pair for pair in pairs(trips_B, trips_A)]

# occupancy data
occ_A = A[['Ritnummer', 'IdDimHalte', 'Bezetting']]
occ_A_dict = occ_A.set_index(['Ritnummer', 'IdDimHalte']).to_dict()['Bezetting']

occ_B = B[['Ritnummer', 'IdDimHalte', 'Bezetting']]
occ_B_dict = occ_B.set_index(['Ritnummer', 'IdDimHalte']).to_dict()['Bezetting']

# trip - exceeding capacity threshold
ex_cap_A = A[['Ritnummer', 'IdDimHalte', 'ex_capacity']]
ex_cap_A_dict = ex_cap_A.set_index(['Ritnummer', 'IdDimHalte']).to_dict()['ex_capacity']

ex_cap_B = B[['Ritnummer', 'IdDimHalte', 'ex_capacity']]
ex_cap_B_dict = ex_cap_B.set_index(['Ritnummer', 'IdDimHalte']).to_dict()['ex_capacity']

''' Calculating waiting time
parameters:
1. headway mean 
2. headway variance '''

h_data = test_data[['Ritnummer', 'PublieksLijnnr', 'dep_datetime', 'RitVertrekTijd']]
h_data = h_data.sort_values('dep_datetime').drop_duplicates(subset=['Ritnummer'], keep='first')
h_data = h_data.sort_values(by=['PublieksLijnnr', 'dep_datetime'], ascending=[False, True])

h_data['h_headway'] = h_data.groupby('PublieksLijnnr')['dep_datetime'].transform(pd.Series.diff)
h_data['h_headway'] = (h_data['h_headway'].dt.total_seconds())/60
h_data['h_headway'] = h_data['h_headway'].fillna(method='bfill')

h_data['h_var'] = h_data.groupby('PublieksLijnnr')['h_headway'].transform(statistics.variance)
h_data['h_mean'] = h_data.groupby('PublieksLijnnr')['h_headway'].transform(statistics.mean)

def cal_waiting_time(mean, var):
   waiting_time = mean * 0.5 + 0.5 * (var / mean)
   return waiting_time

h_data['waiting_time'] = cal_waiting_time(h_data['h_mean'], h_data['h_var']) 
# waiting time as dictionary 
waiting_time_dict = h_data.set_index(['Ritnummer', 'PublieksLijnnr']).to_dict()['waiting_time']



# depature time of trips from the first stop 
dep_time_A = A[['Ritnummer', 'Naam_halte', 'RitVertrekTijd', 'dep_datetime']]
dep_time_A = dep_time_A.sort_values('dep_datetime').drop_duplicates(subset=['Ritnummer'], keep='first')
dep_time_A_dict = dep_time_A.set_index(['Ritnummer', 'Naam_halte']).to_dict()['RitVertrekTijd']

dep_time_B = B[['Ritnummer', 'Naam_halte', 'RitVertrekTijd', 'passeer_datetime']]
dep_time_B = dep_time_B.sort_values('passeer_datetime').drop_duplicates(subset=['Ritnummer'], keep='first')
dep_time_B_dict = dep_time_B.set_index(['Ritnummer', 'Naam_halte']).to_dict()['RitVertrekTijd']


# arrival to the last stop and travel time 
# calculate travel time over each trip 
def cal_travel_time(arr_time, dep_time):
    travel_time = ''
    travel_time = arr_time - dep_time
    return travel_time.dt.total_seconds()/60

''' For later: travel time should be fixed per bus line '''

arr_time_A = A[['Ritnummer', 'Naam_halte', 'dep_datetime', 'passeer_datetime']]
arr_time_A = arr_time_A.loc[arr_time_A.groupby('Ritnummer')['passeer_datetime'].idxmax()]
arr_time_A['travel_time'] = cal_travel_time(arr_time_A['passeer_datetime'], arr_time_A['dep_datetime'])

arr_time_B = B[['Ritnummer', 'Naam_halte', 'dep_datetime', 'passeer_datetime']]
arr_time_B = arr_time_B.loc[arr_time_B.groupby('Ritnummer')['passeer_datetime'].idxmax()]
arr_time_B['travel_time'] = cal_travel_time(arr_time_B['passeer_datetime'], arr_time_B['dep_datetime'])


# calculate deadhead time 


# first and last stop for each trip 
first_last_stop = test_data[['Ritnummer', 'IdDimHalte', 'Naam_halte', 'passeer_datetime', 'dep_datetime']]
first_stop = first_last_stop.loc[first_last_stop.groupby('Ritnummer')['passeer_datetime'].idxmin()]
last_stop = first_last_stop.loc[first_last_stop.groupby('Ritnummer')['passeer_datetime'].idxmax()]


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

model = gp.Model("bus_reassignment")












#%%
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
    date_time = datetime.strptime(date_time, "%y-%m-%d %H:%M:%S").timestamp() * 1000
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

dd = [travel_time[t] for t in travel_time.keys() if t == (dep_stop, arr_stop, 1)]

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
    ("Wesselerbrink", "Wesselerbrink"):0, 
    ("Deppenbroek", "Deppenbroek"):0, 
    ("Glanerbrug", "Glanerbrug"): 0, 
    ("Stroinslanden", "Stroinslanden"):0, 
    ("Zwering", "Zwering"): 0, 
    ("Stokhorst", "Stokhorst"):0, 
    ("Marssteden", "Marssteden"):0, 
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
cost = 40 # euros per minute
# Capacity threshold 
cap_threshold = 60 # in-vehicle crowd

# compute parameters for the model 
num_lines = len(bus_lines)
num_stop = len(bus_stops)
num_trips = len(trips)


# departure time of the first and the last trips 
# convert datetime to millisecond
def conv_time_to_mils(time):
    date = datetime.now().date().strftime("%y-%m-%d ")
    date_time = date + time
    date_time = datetime.strptime(date_time, "%y-%m-%d %H:%M:%S").timestamp() * 1000
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
dep_time = {} # departure time in millisecond
for t in trip:
    for i in bus_lines:
        for j in bus_stops:
            d[t,i,j] = [round(time.time() * 1000)]

a = {} # arrival time in millisecond 
for t in trip:
    for i in L:
        for j in S:
            a[t,i,j] = [round(time.time() * 1000)]

# expected in-vehilce crowding at the most crowded segment
zeta = {} 
for t in trip:
    for i in L:
        for j in S:
            zeta[t,i,j] = [random.randint(0,100)]

# total boarding passengers along all stops within a trip  
theta = {}
for t in trip:
    for i in L:
        for j in S:
            theta[t,i,j] = [random.randint(10,150)]

# maximum capacity threshold 
C = 60 # maximum number of in-vehicle passengers 


# first of the day 
sigma_min = {}
first_trip = datetime.now().date().strftime("%y-%m-%d ") + "05:00:00"
first_trip = datetime.strptime(first_trip, "%y-%m-%d %H:%M:%S").timestamp() * 1000

for i in L:
    for j in S:
        sigma_min[i,j] = [first_trip]
# last trip of the day 
last_trip = datetime.now().date().strftime("%y-%m-%d ") + "23:00:00"
last_trip = datetime.strptime(last_trip, "%y-%m-%d %H:%M:%S").timestamp() * 1000
sigma_max = {}
for i in L:
    for j in S:
        sigma_max[i,j] = [last_trip]

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
    

