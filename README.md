# Bus Reassignment Optimization 

This repository contains code and resources for optimizing bus reassignment in urban transit systems, focusing on mitigating in-vehicle overcrowding during weather-related disruptions. The approach reassign buses dynamically from low-demand trips to lines where the demand is exceeding capacity, minimizing the need for additional buses or drivers.

![optimization](https://github.com/user-attachments/assets/7971b150-181f-41c9-ad0f-573d9cd8d631)

## Data Input

Occupancy Data: Contains boarding and alighting counts for each trip.

Network Graph: Represents the transit network as a directed graph.

Timetable: Includes line details, stop times, and vehicle assignments.

Deadhead Time: Precomputed travel time for buses operating without passengers.

## Reference: 

This project is based on our research:

Farahmand, Z.H., Gkiotsalitis, K., & Geurs, K.T. (2024). Optimal bus reassignment considering in-vehicle overcrowding during weather disruptions. Transportation Research Interdisciplinary Perspectives.
