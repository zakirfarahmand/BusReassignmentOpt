import gurobipy as gp


from gurobipy import GRB
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
    # min sum sum sum (X[t,i,j] - 1) * zeta[t,i,j] * w[t,i,j] + sum sum sum X[t,p,q] theta[t,p,q] w[t,p,q]
    # + sum sum sum X[t,i,j] k[t,q,j] c + sum sum sum X[t,i,j] (lamda[p,q] - lamda[i,j]) c
# CONSTRAINTS:
    # sum X[t,i,j]  - sum X[t,p,q] = 0
    # X[t,i,j] in [0,1]
    #  X[t,i,j] in [0,1]
    #  sigma^min[i,j] < t in rho < sigma^max[i,j]

