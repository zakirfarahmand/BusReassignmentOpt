import gurobipy as gp

from gurobipy import GRB
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

# list of bus stops
S = list([(1,2), (1,3), (1,4)])
d = m.addVars(S, name="d")
m.update()
sum(d.select(1, '*'))
# in tupledict 
d.sum(1, '*')



# list.append()
# list.insert()
# list.pop ==> delete

# using for loop 
a = {0:1, 1:3,2:2}
b = {0:4, 1:5,2:6,3:8}

sum_value = 0 #intially the sum value is zero 

for i in a:
        for j in b:
                sum_value = sum_value + (a[i] + b[j])
print("The sum value is: ", sum_value)

# if statement 
la = ((1,2), (1,3), (2,2))
lb = ((1,2), (1,3), (1,4), (2,1), (2,2), (2,3))
l = {i:4 for i in lb}

print(l)

# sum with if statement 
sum_value = 0;

for i in lb:
        if i not in la:
                sum_value = sum_value + l[i]
        elif i in la:
                sum_value = sum_value - 1
print(sum_value)


#%%
# solving a minimization problem with gurobi 
# obj: \sum \sum cij xij^2
# s.t. \sum xij^2 >= bj, for all j in J
# xij \in Z, forall i in I and j in J
# sets: I = 1,2,3,4 and J = 1,2,3;
# the value of sets does not change, so it is declared as tuples 
I = (1,2,3,4)
J = (1,2,3)
# parameters: bj = {225, 182, 190} and cij = [5 3 2
#                                             7 2 6
#                                             8 3 7
#                                             3 8 13]
c = {(1,1): 5, (1,2): 3, (1,3):2,
        (2,1): 7, (2,2): 2, (2,3): 6,
        (3,1): 8, (3,2): 3, (3,3): 7,
        (4,1): 3, (4,2): 8, (4,3): 13}
bj = {1: 225, 2: 182, 3: 190}

# initiate the model 
model = gp.Model()

# initialize the decision variable xij
x = model.addVars(I,J,vtype=gp.GRB.INTEGER, name="x")
print(x)

# declare objective function 
obj = sum(c[i,j] * x[i,j] * x[i,j] for i in I for j in J)
# add the objective function to the model 
model.setObjective(obj, GRB.MINIMIZE)

# add constraints 
model.addConstrs(quicksum(x[i,j] for i in I) >= bj[j] for j in J)
model.addConstrs(x[i,j] >= 0 for i in I for j in J)

model.optimize()
model.printQuality()

for v in model.getVars():
        print("%s %g" % (v.varName, v.x))
        print("Obj: %g" % model.objVal)
