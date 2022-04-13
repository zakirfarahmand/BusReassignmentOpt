import gurobipy as gp

from gurobipy import * 
import numpy as np
import scipy.sparse as sp
# This example formulates and solves the following simple MIP model :
# maximize
# x + y + 2 z
# subject to
# x + 2 y + 3 z <= 4
# x + y >= 1
# x, y, z binary

try: 
    m = gp.Model("mip1")

    # variables 
    x = m.addVar(vtype=GRB.BINARY, name="x")
    y = m.addVar(vtype=GRB.BINARY, name="y")
    z = m.addVar(vtype=GRB.BINARY, name="z")

    # set objective 
    m.setObjective( x + y + 2 * z, GRB.MAXIMIZE)

    # add constraints
    m.addConstr(x + 2 * y + 3 * z <= 4, "c0")
    m.addConstr(x + y >= 1, "c1")

    # optimize the model 
    m.optimize()

    for v in m.getVars():
        print("%s %g" % (v.varName , v.x))
        print("Obj: %g" % m.objVal)

except gp.GurobiError as e:
        print("Error code: " + str(e.errno) + ":" + str(e))
    
except AttributeError:
        print("Encountered an attribute error")



# Create the model with sparse matrix
m = gp.Model("matrix1")
x = m.addMVar(shape=3, vtype=GRB.BINARY, name="x")
obj = np.array([1.0, 1.0, 2.0])
m.setObjective(obj @ x, GRB.MAXIMIZE)

# constraints
data = np.array([1, 2, 3, -1, -1])
row = np.array([0, 0, 0, 1, 1])
col = np.array([0, 1, 2, 0, 1])

A = sp.csr_matrix((data, (row, col)), shape=(2,3))

# rhs vector 
rhs = np.array([4, -1])

# add constraints
m.addConstr(A @ x <= rhs, name="c")

m.optimize()

print(x.X)
print("Obj: %g" % m.objVal)

values = {}
values['zero'] = 0
values['one'] = 1
values['two'] = 2

print(values['two'])

# multi dictionnary in Gurobi 
names, lower, upper = multidict({'x': [1,2], 'y': [1,2], 'z': [0,3]})
print(upper)

[x*x for x in [1, 2, 3, 4, 5]] 

[(x,y) for x in range(4) for y in range(x+1, 4)]

# tuplelist 
a = tuplelist([(1,2), (1,3), (3,4)])

print(a.select(1, '*'))
print(a.select('*', [3,4]))