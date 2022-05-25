

def selected(vals):
    s = {}
    for i, j in vals.keys():
        if vals[i, j] > 0.5:
            s.append((i, j))
    return s


''' Comments:
1. some trips on line 9 (Enschede - Hengelo) do not start from Enschede central station
2. some trips on line 1 (De Posten - UT)
3. Bezetting increases/decreases without changes in the number instappers/outstappers'''

# %%
'''
Model parameters
 '''


# occupancy data
occ_A = A[['Ritnummer', 'IdDimHalte', 'Bezetting']]
occ_A_dict = occ_A.set_index(
    ['Ritnummer', 'IdDimHalte']).to_dict()['Bezetting']

occ_B = B[['Ritnummer', 'IdDimHalte', 'Bezetting']]
occ_B_dict = occ_B.set_index(
    ['Ritnummer', 'IdDimHalte']).to_dict()['Bezetting']

# trip - exceeding capacity threshold
ex_cap_A = A[['Ritnummer', 'IdDimHalte', 'ex_capacity']]
ex_cap_A_dict = ex_cap_A.set_index(
    ['Ritnummer', 'IdDimHalte']).to_dict()['ex_capacity']

ex_cap_B = B[['Ritnummer', 'IdDimHalte', 'ex_capacity']]
ex_cap_B_dict = ex_cap_B.set_index(
    ['Ritnummer', 'IdDimHalte']).to_dict()['ex_capacity']


# calculate deadhead time


# first and last stop for each trip
first_last_stop = test_data[['Ritnummer', 'IdDimHalte',
                             'Naam_halte', 'passeer_datetime', 'dep_datetime']]
first_stop = first_last_stop.loc[first_last_stop.groupby(
    'Ritnummer')['passeer_datetime'].idxmin()]
last_stop = first_last_stop.loc[first_last_stop.groupby(
    'Ritnummer')['passeer_datetime'].idxmax()]


# deadhead cost
# trip arc and travel time between the first and last stop of each line
arc, travel_time = gp.multidict({
    ("ECS", "UT", 1): 35,
    ("UT", "ECS", 1): 32,
    ("ECS", "Wesselerbrink", 1): 25,
    ("Wesselerbrink", "ECS", 1): 32,
    ("ECS", "Deppenbroek", 2): 35,
    ("Deppenbroek", "ECS", 2): 40,
    ("ECS", "Glanerbrug", 3): 50,
    ("Glanerbrug", "ECS", 3): 45,
    ("ECS", "Stroinslanden", 4): 28,
    ("Stroinslanden", "ECS", 4): 33,
    ("ECS", "Zwering", 5): 41,
    ("Zwering", "ECS", 5): 39,
    ("ECS", "Stokhorst", 6): 32,
    ("Stokhorst", "ECS", 6): 28,
    ("ECS", "Marssteden", 7): 42,
    ("Marssteden", "ECS", 7): 48,
    ("ECS", "Hengelo", 9): 35,
    ("Hengelo", "ECS", 9): 35
})


grouped_data = pd.DataFrame(test_data.groupby(['PublieksLijnnr', 'Ritnummer'])[
    'dep_datetime', 'IdDimHalte', 'Naam_halte'].first().reset_index())
grouped_data['dep_datetime'] = grouped_data['dep_datetime'].apply(
    conv_time_to_mils)

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")
# ax.scatter(grouped_data['PublieksLijnnr'], grouped_data['IdDimHalte'], grouped_data['dep_datetime'], c='b')
for i in range(len(grouped_data)):
    x = grouped_data.loc[i, 'PublieksLijnnr']
    y = grouped_data.loc[i, 'IdDimHalte']
    z = grouped_data.loc[i, 'dep_datetime']
    label = grouped_data.loc[i, 'Naam_halte']
    ax.scatter(x, y, z, color='b')
    ax.text(x, y, z, '%s' % (label), size=8, color='k')
ax.set_xlabel('Bus line')
ax.set_ylabel('Bus stop')
ax.set_zlabel('Departure time')
plt.show()


# Writing mathematical formulation
"""
Objective 1: minimize waiting time of stranded passengers (re-assignment vs cancellation)

objective 2: minimize deadhead cost (from the last stop of timetabled trip to the first stop of re-assigned trip and back to the first stop of following trip operated by the same bus)

Constratins:
   1. arrival time to the first stop of re-assigned trip <= depature of the crowded trip
   2. Re-assignment: a cancelled trip can be only re-assigned once and only one bus trip is re-assigned before an overcrowded trip
   3. imposed cancellation: If the bus operating the cancelled trip cannot arrive for its next trip, its following trip will also be cancelled.
   4. max imposed cancellation is 2

"""

# %%
# calculate the pairwise potential re-assignment matrix

for t in range(len(dep_time_B)):
    l = [dep_time_B.iloc[t]['Ritnummer'],
         dep_time_B.iloc[t]['passeer_datetime']]
    multi[t] = l
key, trip, dep_time = gp.multidict(multi)


# %%
bus_stops = ['ECS', "UT", "Wesselerbrink", "Deppenbroek", "Glanerbrug", "Stroinslanden",
             "Zwering", "Stokhorst", "Marssteden", "Hengelo"]

# set of trip number
# later this set should be based on the actual data
# each trip has number, first stop, last stop and line number
trip, dep_stop, arr_stop, line_number = gp.multidict({
    1: ["ECS", "UT", 1],
    2: ["UT", "ECS", 1],
    3: ["ECS", "Wesselerbrink", 1],
    4: ["De_Posten", "ECS", 1],
    5: ["ECS", "Deppenbroek", 2],
    6: ["Deppenbroek", "ECS", 2],
    7: ["ECS", "Glanerbrug", 3],
    8: ["Glanerbrug", "ECS", 3],
    9: ["ECS", "Stroinslanden", 4],
    10: ["Stroinslanden", "ECS", 4],
    11: ["ECS", "Zwering", 5],
    12: ["Zwering", "ECS", 5],
    13: ["ECS", "Stokhorst", 6],
    14: ["Stokhorst", "ECS", 6],
    15: ["ECS", "Marssteden", 7],
    16: ["Marssteden", "ECS", 7],
    17: ["ECS", "Hengelo", 9],
    18: ["Hengelo", "ECS", 9]
})

# each trip has departure time from the first stop
# departure time trips
# convert datetime to millisecond


def conv_time_to_mils(time):
    date = datetime.now().date().strftime("%y-%m-%d ")
    date_time = date + time
    date_time = datetime.strptime(
        date_time, "%y-%m-%d %H:%M:%S").timestamp() * 1000
    return date_time


trip, dep_time = gp.multidict({
    1: conv_time_to_mils("09:12:00"),
    2: conv_time_to_mils("09:15:00"),
    3: conv_time_to_mils("09:20:00"),
    4: conv_time_to_mils("09:25:00"),
    5: conv_time_to_mils("09:30:00"),
    6: conv_time_to_mils("09:35:00"),
    7: conv_time_to_mils("09:15:00"),
    8: conv_time_to_mils("09:36:00"),
    9: conv_time_to_mils("09:20:00"),
    10: conv_time_to_mils("09:25:00"),
    11: conv_time_to_mils("09:40:00"),
    12: conv_time_to_mils("09:25:00"),
    13: conv_time_to_mils("09:25:00"),
    14: conv_time_to_mils("09:40:00"),
    15: conv_time_to_mils("09:10:00"),
    16: conv_time_to_mils("09:20:00"),
    17: conv_time_to_mils("09:00:00"),
    18: conv_time_to_mils("09:15:00")
})

# trip arc and travel time between the first and last stop of each line
arc, travel_time = gp.multidict({
    ("ECS", "UT", 1): 35,
    ("UT", "ECS", 1): 32,
    ("ECS", "Wesselerbrink", 1): 25,
    ("Wesselerbrink", "ECS", 1): 32,
    ("ECS", "Deppenbroek", 2): 35,
    ("Deppenbroek", "ECS", 2): 40,
    ("ECS", "Glanerbrug", 3): 50,
    ("Glanerbrug", "ECS", 3): 45,
    ("ECS", "Stroinslanden", 4): 28,
    ("Stroinslanden", "ECS", 4): 33,
    ("ECS", "Zwering", 5): 41,
    ("Zwering", "ECS", 5): 39,
    ("ECS", "Stokhorst", 6): 32,
    ("Stokhorst", "ECS", 6): 28,
    ("ECS", "Marssteden", 7): 42,
    ("Marssteden", "ECS", 7): 48,
    ("ECS", "Hengelo", 9): 35,
    ("Hengelo", "ECS", 9): 35
})
# calculate arrival time


def cal_arr_time(dep_time, travel_time):
    return dep_time + travel_time * 60000


dd = [travel_time[t]
      for t in travel_time.keys() if t == (dep_stop, arr_stop, 1)]

for i in dep_stop.keys():
    key = (dep_stop[i], arr_stop[i], line_number[i])
    if key == ('Zwering', 'ECS', 5):
        print(key)
tt = [t for t in trip]


trip, arr_time = gp.multidict({


    1: cal_arr_time(dep_time[0], [a]),
    2: conv_time_to_mils("09:15:00"),
    3: conv_time_to_mils("09:20:00"),
    4: conv_time_to_mils("09:25:00"),
    5: conv_time_to_mils("09:30:00"),
    6: conv_time_to_mils("09:35:00"),
    7: conv_time_to_mils("09:15:00"),
    8: conv_time_to_mils("09:36:00"),
    9: conv_time_to_mils("09:20:00"),
    10: conv_time_to_mils("09:25:00"),
    11: conv_time_to_mils("09:40:00"),
    12: conv_time_to_mils("09:25:00"),
    13: conv_time_to_mils("09:25:00"),
    14: conv_time_to_mils("09:40:00"),
    15: conv_time_to_mils("09:10:00"),
    16: conv_time_to_mils("09:20:00"),
    17: conv_time_to_mils("09:00:00"),
    18: conv_time_to_mils("09:15:00")
})

# combine two dictionaries
for key, value in dep_stop.items():
    if key in dep_time.keys():
        print(key, value, dep_time[key])

for t in trip:
    return dep_time[t] +


# arc and deadhead time
darc, deadhead_time = gp.multidict({
    ("ECS", "ECS"): 0,
    ("UT", "UT"): 0,
    ("Wesselerbrink", "Wesselerbrink"): 0,
    ("Deppenbroek", "Deppenbroek"): 0,
    ("Glanerbrug", "Glanerbrug"): 0,
    ("Stroinslanden", "Stroinslanden"): 0,
    ("Zwering", "Zwering"): 0,
    ("Stokhorst", "Stokhorst"): 0,
    ("Marssteden", "Marssteden"): 0,
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

# driving cost per minute
cost = 40  # euros per minute
# Capacity threshold
cap_threshold = 60  # in-vehicle crowd

# compute parameters for the model
num_lines = len(bus_lines)
num_stop = len(bus_stops)
num_trips = len(trips)


# departure time of the first and the last trips
# convert datetime to millisecond
def conv_time_to_mils(time):
    date = datetime.now().date().strftime("%y-%m-%d ")
    date_time = date + time
    date_time = datetime.strptime(
        date_time, "%y-%m-%d %H:%M:%S").timestamp() * 1000
    return date_time


arc, trip, dep_time = gp.multidict({
    ("ECS", "UT"): [1, conv_time_to_mils("09:12:00")],
    ("UT", "ECS"): [2, conv_time_to_mils("09:15:00")],
    ("ECS", "Wesselerbrink"): [3, conv_time_to_mils("09:20:00")],
    ("Wesselerbrink", "ECS"): [4, conv_time_to_mils("09:25:00")],
    ("ECS", "Deppenbroek"): [5, conv_time_to_mils("09:30:00")],
    ("Deppenbroek", "ECS"): [6, conv_time_to_mils("09:35:00")],
    ("ECS", "Glanerbrug"): [7, conv_time_to_mils("09:15:00")],
    ("Glanerbrug", "ECS"): [8, conv_time_to_mils("09:36:00")],
    ("ECS", "Stroinslanden"): [9, conv_time_to_mils("09:20:00")],
    ("Stroinslanden", "ECS"): [10, conv_time_to_mils("09:25:00")],
    ("ECS", "Zwering"): [11, conv_time_to_mils("09:40:00")],
    ("Zwering", "ECS"): [12, conv_time_to_mils("09:25:00")],
    ("ECS", "Stokhorst"): [13, conv_time_to_mils("09:25:00")],
    ("Stokhorst", "ECS"): [14, conv_time_to_mils("09:40:00")],
    ("ECS", "Marssteden"): [15, conv_time_to_mils("09:10:00")],
    ("Marssteden", "ECS"): [16, conv_time_to_mils("09:20:00")],
    ("ECS", "Hengelo"): [17, conv_time_to_mils("09:00:00")]
})


# for each trip, there are three identicators line number, starting stop, and depature time/arrival time
trip = [t for t in trips]
dep_time = {}  # departure time in millisecond
for t in trip:
    for i in bus_lines:
        for j in bus_stops:
            d[t, i, j] = [round(time.time() * 1000)]

a = {}  # arrival time in millisecond
for t in trip:
    for i in L:
        for j in S:
            a[t, i, j] = [round(time.time() * 1000)]

# expected in-vehilce crowding at the most crowded segment
zeta = {}
for t in trip:
    for i in L:
        for j in S:
            zeta[t, i, j] = [random.randint(0, 100)]

# total boarding passengers along all stops within a trip
theta = {}
for t in trip:
    for i in L:
        for j in S:
            theta[t, i, j] = [random.randint(10, 150)]

# maximum capacity threshold
C = 60  # maximum number of in-vehicle passengers


# first of the day
sigma_min = {}
first_trip = datetime.now().date().strftime("%y-%m-%d ") + "05:00:00"
first_trip = datetime.strptime(
    first_trip, "%y-%m-%d %H:%M:%S").timestamp() * 1000

for i in L:
    for j in S:
        sigma_min[i, j] = [first_trip]
# last trip of the day
last_trip = datetime.now().date().strftime("%y-%m-%d ") + "23:00:00"
last_trip = datetime.strptime(
    last_trip, "%y-%m-%d %H:%M:%S").timestamp() * 1000
sigma_max = {}
for i in L:
    for j in S:
        sigma_max[i, j] = [last_trip]

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


# model without function
model = gp.Model("Bus Reassignment")
# decision variable
# potential reassignment variable = 1, if i could be re-assigned before j
# create pair of potential trips for re-assignemnt and trips that they could be re-assigned before
paired_trips = tuplelist()
for i in reAssign:
    for j in toAssign:
        if arr_time_dict[i] <= dep_time_dict[j]:
            paired_trips += [(i, j)]

reassign_var = model.addVars(
    paired_trips, vtype=GRB.BINARY, name='reassign_var')
model.update()

# potential imposed cancellation variable = 1 if k is cancelled due to reassignment for j

# there should be at least one trip assigned before overcrowded trip
model.addConstrs((reassign_var.sum('*', j) <=
                 1 for j in toAssign), name='reassignment')
# a trip can only be reassigned once
model.addConstrs((reassign_var.sum(i, '*') <=
                 1 for i in reAssign), name='cancellation')

model.update()

obj = quicksum(0.5 * reassign_var[i, j] * ex_capacity_dict[j, s] * waiting_time_dict[j] for i, j in paired_trips for s in stops_dict[j]) + quicksum(3 * (1-reassign_var[i, j]) * ex_capacity_dict[j, s]
                                                                                                                                                    * waiting_time_dict[j] for i, j in paired_trips for s in stops_dict[j]) + quicksum(2 * reassign_var[i, j] * demand_dict[i, s] * waiting_time_dict[i] for i, j in paired_trips for s in stops_dict[i])

model.setObjective(obj, GRB.MINIMIZE)
model.update()

model.optimize()
