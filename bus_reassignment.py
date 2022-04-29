import gurobipy as gp


from gurobipy import GRB
from gurobipy import multidict, tuplelist, quicksum
import numpy as np
import scipy.sparse as sp

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
# PARAMETERS:
