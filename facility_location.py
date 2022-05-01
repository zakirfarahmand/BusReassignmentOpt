from itertools import product
from math import sqrt
import gurobipy as gp
from gurobipy import GRB

# parameters 
customers = [(0,1.5), (2.5, 1.2)]
facilities = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
setup_cost = [3,2,3,1,3,3,4,3,2]
cost_per_mile = 1

#%%
"""
Preprocessing 
""" 
# Euclidian distance between facilities and customer sites 
def compute_distance(loc1,loc2):
    dx = loc1[0] - loc2[0]
    dy = loc1[1] - loc2[1]
    return sqrt(dx*dx + dy*dy)
    
    
# computer parameter for MIP model formulation 
num_facilities = len(facilities)
num_customers = len(customers)
cartesian_prod = list(product(range(num_customers), range(num_facilities)))

# shipping cost 
shipping_cost = {(c,f): cost_per_mile * compute_distance(customers[c], facilities[f]) for c, f in cartesian_prod}

#%%
# Model Deployment

model = gp.Model('facility_location')
# decision variables 
select = model.addVars(num_facilities, vtype=GRB.BINARY, name="select")
assign = model.addVars(cartesian_prod, ub=1, vtype=GRB.CONTINUOUS, name='Assign')

# constraints 
model.addConstrs((assign[(c,f)] <= select[f] for c,f in cartesian_prod), name="setup2ship")
model.addConstrs((gp.quicksum(assign[(c,f)] for f in range(num_facilities)) == 1 for c in range(num_customers)), name='Demand')
model.setObjective(select.prod(setup_cost) + assign.prod(shipping_cost), GRB.MINIMIZE)

model.optimize()

# %%
# analyze the solution 
# display which option is the optimal one 
for facility in select.keys():
    if (abs(select[facility].x) > 1e-6):
        print(f"\n Build a warehouse at location {facility + 1}.")

# shipment cost from facility to customers 
for customer, facility in assign.keys():
    if (abs(assign[customer, facility].x) > 1e-6):
        print(f"\n Supermarket {customer +1} receives {round(100*assign[customer, facility].x, 2)} % of its demand from Warehouse {facility + 1}.")