
# Bus re-assignment problem description
# SETS:
    #   T = [1,2,..., t] set of bus trips 
    #   S_t = [1, 2,..., s_t] set of bus stops served by trip t
    #   T_a = subset of trips where the new bus trips will be re-assigned 
    #   A_r = subset of trips where the bus trips will be canceled. 
    #   F_j = set of following trips for trip j operated by the same bus 
    #   A = a matrix with T_a columns and T_r rows 

# PARAMETERS:
    # dep[i] => departure time of trip i \in T
    # dep[j] => departure time of trip j
    # deadhead_time[i,j] => deadhead time from the last stop of j to the first stop of i
    # travel_time[i] => average travel time for trip i \in T
    # C => driving cost per minute 
    # zeta[t,i,j] => the expected in-vehicle crowding at the most crowded segment of line i for trip t dispatching from stop j
    # theta[t,i,j] => the expected boarding passengers along all stops of line i for trip t dispatching from stop j
    # w[t,i,j] => average passenger waiting time during trip t 
    # C => capacity threshold 
    # sigma^min[i,j] => dispatching time of the first trip 
    # sigma^max[i,j] => dispatching time of the last trip 
# DECISION VARIABLES
    # X[t,i,j] = the re-assignment variable 
    # X[t,p,q] = the cancellation variable

# objective function: 
    # min \sum \sum \sum (X[t,i,j] - 1) * zeta[t,i,j] * w[t,i,j] + \sum \sum \sum X[t,p,q] theta[t,p,q] w[t,p,q]
    # + \sum \sum \sum X[t,i,j] k[t,q,j] c + \sum \sum \sum X[t,i,j] (lamda[p,q] - lamda[i,j]) c
# CONSTRAINTS:
    # \sum X[t,i,j]  - \sum X[t,p,q] = 0
    # X[t,i,j] in [0,1]
    #  X[t,i,j] in [0,1]
    #  sigma^min[i,j] < t in rho < sigma^max[i,j]

import gurobipy as gp
from gurobipy import GRB
from gurobipy import multidict, tuplelist, quicksum
import numpy as np
import pandas as pd 
import scipy.sparse as sp
import time
from datetime import datetime
import random
# SETS:

T = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]


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
    

