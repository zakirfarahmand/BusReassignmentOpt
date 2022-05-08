# In this problem, we have six end customers, each with a known demand for a product.  Customer demand can be satisfied from a set of four depots, or directly from a set of two factories.  Each depot can support a maximum volume of product moving through it, and each factory can produce a maximum amount of product.  There are known costs associated with transporting the product, from a factory to a depot, from a depot to a customer, or from a factory directly to a customer.


# SETS
#   f in Factories {Liverpool, Brighton}
#   d in Depots {Newcastle, Birmingham, London, Exeter}
#   c in Customers {c1, c2, c3, c4, c5, c6}

# PARAMETERS
#   cost s,t = shipping cost from s to t 
#   supply f = max supply from factory f 
#   through d = max product flow through depot d 
#   demand c = demand for goods from customer c 

# DECISION VARIABLES 
# flow s,t = amount of product shiped from s to t 

# constraints:
# sum flow f, t <= supply f 
# sum flow s,c <= demand c 
# sum flow s,d <= flow d,t
# sum flow s,d <= through d 

import pandas as pd
import gurobipy as gp 
from gurobipy import GRB 

#%% 
# create input data list 
supply = dict({'Liverpool': 150000,
               'Brighton': 200000})

through = dict({'Newcastle': 70000,
                'Birmingham': 50000,
                'London': 100000,
                'Exeter': 40000})

demand = dict({'C1': 50000,
               'C2': 10000,
               'C3': 40000,
               'C4': 35000,
               'C5': 60000,
               'C6': 20000})

# Create a dictionary to capture shipping costs.

arcs, cost = gp.multidict({
    ('Liverpool', 'Newcastle'): 0.5,
    ('Liverpool', 'Birmingham'): 0.5,
    ('Liverpool', 'London'): 1.0,
    ('Liverpool', 'Exeter'): 0.2,
    ('Liverpool', 'C1'): 1.0,
    ('Liverpool', 'C3'): 1.5,
    ('Liverpool', 'C4'): 2.0,
    ('Liverpool', 'C6'): 1.0,
    ('Brighton', 'Birmingham'): 0.3,
    ('Brighton', 'London'): 0.5,
    ('Brighton', 'Exeter'): 0.2,
    ('Brighton', 'C1'): 2.0,
    ('Newcastle', 'C2'): 1.5,
    ('Newcastle', 'C3'): 0.5,
    ('Newcastle', 'C5'): 1.5,
    ('Newcastle', 'C6'): 1.0,
    ('Birmingham', 'C1'): 1.0,
    ('Birmingham', 'C2'): 0.5,
    ('Birmingham', 'C3'): 0.5,
    ('Birmingham', 'C4'): 1.0,
    ('Birmingham', 'C5'): 0.5,
    ('London', 'C2'): 1.5,
    ('London', 'C3'): 2.0,
    ('London', 'C5'): 0.5,
    ('London', 'C6'): 1.5,
    ('Exeter', 'C3'): 0.2,
    ('Exeter', 'C4'): 1.5,
    ('Exeter', 'C5'): 0.5,
    ('Exeter', 'C6'): 1.5
})

# deploy the model 
model = gp.Model("supply_network_design")
flow = model.addVars(arcs, obj=cost, name='flow')
 
# constraints 
# factory flow 
factories = supply.keys()
factory_flow = model.addConstrs((gp.quicksum(flow.select(factory, '*')) <= supply[factory] for factory in factories), name="factory")

# customer demand 
customers = demand.keys()
customer_flow = model.addConstrs((gp.quicksum(flow.select("*", customer)) == demand[customer] for customer in customers), name="customer")

# depot flow 
depots = through.keys()
depot_flow = model.addConstrs((gp.quicksum(flow.select(depot, '*')) == gp.quicksum(flow.select("*", depot)) for depot in depots), name="depot")

# depot through 
depot_capacity = model.addConstrs((gp.quicksum(flow.select("*", depot)) <= through[depot] for depot in depots), name="depot_capacity")

# model optimization 
model.optimize()

#%%
# analysis of results 
product_flow = pd.DataFrame(columns=["From", "To", "Flow"])
for arc in arcs:
    if flow[arc].x > 1e-6:
        product_flow = product_flow.append({"From": arc[0], "To": arc[1], "Flow": flow[arc].x}, ignore_index=True)
product_flow.index = [""]*len(product_flow)
product_flow