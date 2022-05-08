# The Customer Assignment Problem is closely related to the Facility Location Problem, which is concerned with the optimal placement of facilities (from a set of candidate locations) in order to minimize the distance between the company's facilities and the customers. When the facilities have unlimited capacity, customers are assumed to be served by the closest facility.

# In cases where the number of customers considered is too big, the customers can be grouped into clusters. Then, the cluster centers can be used in lieu of the individual customer locations. This pre-processing makes the assumption that all customers belonging to a given cluster will be served by the facility assigned to that cluster. The k-means algorithm can be used for this task, which aims to partition n objects into k distinct and non-overlapping clusters.

import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import random

seed = 10101
num_customers = 500000
num_candidates = 20 
max_facilities = 8
num_clusters = 50 
num_gaussians = 10 
threshold = 0.99

random.seed(seed)
customers_per_gaussian = np.random.multinomial(num_customers, [1/num_gaussians] * num_gaussians)

customer_locs = []
for i in range(num_gaussians):
    center = (random.random()-0.5, random.random()-0.5)
    customer_locs += [(random.gauss(0,.1) + center[0], random.gauss(0,.1)+center[1]) for i in range(customers_per_gaussian[i])]

facility_locs = [(random.random()-0.5, random.random()-0.5) for i in range(num_candidates)] 

print("First customer location: ", customer_locs[0])

#%% 
# Clustering 
kmeans = MiniBatchKMeans(n_clusters=num_clusters, init_size=3*num_clusters,
                         random_state=seed).fit(customer_locs)
memberships = list(kmeans.labels_)
centroids = list(kmeans.cluster_centers_)
weights = list(np.histogram(memberships, bins=num_clusters)[0])
print('First cluster center:', centroids[0])
print('Weights for first 10 clusters:', weights[:10])


def dist(loc1, loc2):
    return np.linalg.norm(loc1-loc2, ord=2) # Euclidean distance

pairings = {(facility, cluster): dist(facility_locs[facility], centroids[cluster])
            for facility in range(num_candidates)
            for cluster in range(num_clusters) 
            if  dist(facility_locs[facility], centroids[cluster]) < threshold}
print("Number of viable pairings: {0}".format(len(pairings.keys())))
# %%
# model deployment 
model = gp.Model("facility_location")

# decesion variables 
select = model.addVars(range(num_candidates), vtype=GRB.BINARY, name="select")
assign = model.addVars(pairings.keys(), vtype=GRB.BINARY, name="assign")

# objective function 
obj = gp.quicksum(weights[cluster]
               *pairings[facility, cluster]
               *assign[facility, cluster]
               for facility, cluster in pairings.keys())
model.setObjective(obj, GRB.MINIMIZE)

model.setObjective(obj, GRB.MINIMIZE)

# ADD CONSTRAINTS
# facility limit 
model.addConstr(select.sum() <= max_facilities, name="Facility_limit")
# open to assign 
model.addConstrs((assign[facility, cluster] <= select[facility] for facility, cluster in pairings.keys()), name="open2assign") 
# closest store
model.addConstrs((assign.sum("*", cluster) ==1 for cluster in range(num_clusters)), name="closest_store")

# find optimal solution 
model.optimize()


plt.figure(figsize=(8,8), dpi=150)
plt.scatter(*zip(*customer_locs), c='Pink', s=0.5)
plt.scatter(*zip(*centroids), c='Red', s=10)
plt.scatter(*zip(*facility_locs), c='Green', s=10)
assignments = [p for p in pairings if assign[p].x > 0.5]
for p in assignments:
    pts = [facility_locs[p[0]], centroids[p[1]]]
    plt.plot(*zip(*pts), c='Black', linewidth=0.1)