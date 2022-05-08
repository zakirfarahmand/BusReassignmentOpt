import gurobipy as gp
from gurobipy import GRB

# import datasets 
commodities = ['Pencils', 'Pens']; 
nodes = ['Dentroit', 'Denver', 'Boston', 'New York', 'Seattle']; 

# create tranport arc 
# each arc has two identicators: origin and destination and a value 
arcs, capacity = gp.multidict({
    ('Dentriot', 'Boston'): 100,
    ('Detroit', 'New York'): 80, 
    ('Detriot', 'Seattle'): 120, 
    ('Denver', 'Boston'): 120,
    ('Denver', 'New York'): 120,
    ('Denver', 'Seattle'): 120
})

# cost of transporting each commodity from origin to destination
cost = {
    ('Pencils', 'Detroit', 'Boston'): 10, 
    ('Pencils', 'Detroit', 'New York'): 20, 
    ('Pencils', 'Detroit', 'Seattle'): 60, 
    ('Pencils', 'Denver', 'Boston'): 40, 
    ('Pencils', 'Denver', 'New York'): 40, 
    ('Pencils', 'Denver', 'Seattle'): 30, 
    ('Pens', 'Detroit', 'Boston'): 20, 
    ('Pens', 'Detroit', 'New York'): 20, 
    ('Pens', 'Detroit', 'Seattle'): 80, 
    ('Pens', 'Denver', 'Boston'): 60, 
    ('Pens', 'Denver', 'New York'): 70, 
    ('Pens', 'Denver', 'Seattle'): 30
    
}

# demand for pair of pens/pencils and cities 
inflow = {
    ('Pencils', 'Detriot'): 50,
    ('Pencils', 'Denver'): 60,
    ('Pencils', 'Boston'): -50,
    ('Pencils', 'New York'): -50, 
    ('Pencils', 'Seattle'): -10, 
    ('Pens', 'Detriot'): 60,
    ('Pens', 'Denver'): 40,
    ('Pens', 'Boston'): -40,
    ('Pens', 'New York'): -30,
    ('Pens', 'Seattle'): -30
}; 

# create the optimization model 
m = gp.Model('netflow')

# create decision variables 
flow = m.addVars(commodities, arcs, obj=cost, name="flow")
# constraints for capacity 
m.addConstrs((flow.sum('*', i, j) <= capacity[i,j] for i, j in arcs), 'cap')

# constraints for flow conservation
m.addConstrs(
    (flow.sum())
)