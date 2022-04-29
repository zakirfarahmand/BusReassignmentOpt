import gurobipy as gp


from gurobipy import GRB
from gurobipy import multidict, tuplelist, quicksum
import numpy as np
import scipy.sparse as sp
import time
from datetime import datetime
import random
# Bus re-assignment problem description
# SETS:
    #   L = [1,2,3,4,5,6,7,8,9] set of bus lines
    #   S = ['EnschedeCS', "UT", "Wesslerbrink", "Deppenbroek", "Glanerbrug",   
    #       "Stroinslanden", "Zwering", "Stokhorst", "Marssteden",
    #       "Hengelo"] set of bus stops
    #  rho = [[1, 2, 3, 4, 5, 6, 7, 
    #           8, 9, 10, 11, 12, 13]] set timetabled trips 
    #   Ab = subset of rho 
    #   Al = subset of rho 

# PARAMETERS:
    # d[t,i,j] => departure time of trip t dispatching from stop j of line i in millisecond
    # a[t,i,j] => arrival time of trip t to stop j of line i in millisecond
    # k[q,j] => deadhead time between two stops q and j in minute
    # lamda[i,j] => average travel time for a bus dispatching from stop j of line i in minute
    # c => driving cost per minute 
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

# SETS:
L = [1,2,3,4,5,6,7,8,9]
S = ['ECS', "UT", "Wesselerbrink", "Deppenbroek", "Glanerbrug", "Stroinslanden",
     "Zwering", "Stokhorst", "Marssteden", "Hengelo"]
rho = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]


# trip arc and travel time in minute
arc, line_number = gp.multidict({
    ("ECS", "UT"): 1, 
    ("UT", "ECS"): 1,
    ("ECS", "Wesselerbrink"): 1,
    ("Wesselerbrink", "ECS"): 1, 
    ("ECS", "Deppenbroek"): 2,
    ("Deppenbroek", "ECS"): 2,
    ("ECS", "Glanerbrug"): 3, 
    ("Glanerbrug", "ECS"): 3,
    ("ECS", "Stroinslanden"): 4, 
    ("Stroinslanden", "ECS"): 4,
    ("ECS", "Zwering"): 5, 
    ("Zwering", "ECS"): 5,
    ("ECS", "Stokhorst"): 6,
    ("Stokhorst", "ECS"): 6,
    ("ECS", "Marssteden"): 8,
    ("Marssteden", "ECS"): 8, 
    ("ECS", "Hengelo"): 9
})

# trip arc and travel time in minute
arc, travel_time = gp.multidict({
    ("ECS", "UT"): 35, 
    ("UT", "ECS"): 32,
    ("ECS", "Wesselerbrink"): 25,
    ("Wesselerbrink", "ECS"): 32, 
    ("ECS", "Deppenbroek"): 35,
    ("Deppenbroek", "ECS"): 40,
    ("ECS", "Glanerbrug"): 50, 
    ("Glanerbrug", "ECS"): 45,
    ("ECS", "Stroinslanden"): 28, 
    ("Stroinslanden", "ECS"): 33,
    ("ECS", "Zwering"): 41, 
    ("Zwering", "ECS"): 39,
    ("ECS", "Stokhorst"): 32,
    ("Stokhorst", "ECS"): 28,
    ("ECS", "Marssteden"): 42,
    ("Marssteden", "ECS"): 48, 
    ("ECS", "Hengelo"): 35
})

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

# for each trip, there are three identicators line number, starting stop, and depature time/arrival time
trip = [t for t in rho]
d = {} # departure time in millisecond
for t in trip:
    for i in L:
        for j in S:
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
# operating cost per minute 
c = 100 # euros per minute 

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
    

