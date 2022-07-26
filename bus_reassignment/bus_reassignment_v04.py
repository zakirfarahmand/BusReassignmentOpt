# import libraries
from soupsieve import select
import data_preprocessing as dp
import googlemaps
import gurobipy as gp
from gurobipy import GRB
from gurobipy import multidict, tuplelist, quicksum
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time
from datetime import datetime
import random
import pyodbc
import datetime as dt
from datetime import timedelta, datetime
from itertools import combinations, product
import statistics
from matplotlib import animation
from IPython.display import HTML
from termcolor import colored
import googlemaps
import sys
import pytz
gmaps = googlemaps.Client(key='AIzaSyAra0o3L3rs-uHn4EpaXx1Y57SIF_02684')

# AIzaSyBmoRMRmfpYO5AG3GsSbmKlyNPaWOkKROE
# import functions
# insert year, month, and day

date = '2022-02-11'
data = dp.import_data(date)


def deadhead():
    cursor, conn = dp.connect_to_database()
    deadhead_time = pd.read_sql_query('select * from deadhead_time', conn)
    deadhead_dict = {}

    cursor.close()
    deadhead_dict.update({(i, j): k for i, j, k in zip(
        deadhead_time.stopA, deadhead_time.stopB, deadhead_time.deadhead)})

    return deadhead_dict

# %%


def conv_time_to_mils(date_time):
    return date_time.timestamp() * 1000

# %%


''' list of to assign and re-assign should only from Enschede bus lines '''


def sets_parameters(data):
    capacity_threshold = 35
    demand_dict = {}  # number of expected boarding passengers at every stop
    # number of passengers who cannot board buses due to overcrowding
    stranded_passenger_dict = {}
    stops_dict = {}  # list of all stops along each trip
    trip_following_trips = {}
    bus_trips = {}  # list trips operated by the same bus
    line_trips = {}  # list of trips on the same bus line
    # convert timestamp into milliseconds
    # data['DepartureTime'] = data['DepartureTime'].apply(
    #     conv_time_to_mils)

    # remove the first and last trip of the day
    first_trip = data.sort_values(by=['Systeemlijnnr', 'DepartureTime']).drop_duplicates(
        subset=['TripNumber'], keep='first')
    first_trip = first_trip.loc[first_trip.groupby(
        by=['Systeemlijnnr', 'Direction']).DepartureTime.idxmin()]
    first_trip = first_trip['TripNumber'].tolist()
    last_trip = data.sort_values(by=['Systeemlijnnr', 'DepartureTime']).drop_duplicates(
        subset=['TripNumber'], keep='last')
    last_trip = last_trip.loc[last_trip.groupby(
        by=['Systeemlijnnr', 'Direction']).DepartureTime.idxmax()]
    last_trip = last_trip['TripNumber'].tolist()
    data = data[~data['TripNumber'].isin(first_trip)]
    data = data[~data['TripNumber'].isin(last_trip)]

    # calculate occupancy exceeding the capacity threshold
    data['exceeding_capacity'] = data['OccupancyCorrected'].apply(
        lambda x: x - capacity_threshold if (x > capacity_threshold) else 0)

    demand_dict = {(i, j): k for i, j, k in zip(
        data['TripNumber'], data['IdDimHalte'], data['OccupancyCorrected'])}

    stranded_passenger_dict = data.sort_values(by=['TripNumber', 'IdDimHalte']).set_index(
        ['TripNumber', 'IdDimHalte']).to_dict()['exceeding_capacity']

    stops_dict = {k: list(v)
                  for k, v in data.groupby('TripNumber')['IdDimHalte']}

    enschede_lines = [4701, 4702, 4703, 4704, 4705, 4706, 4707, 4709]
    enschede = data[data['Systeemlijnnr'].isin(enschede_lines)]
    # list of trips exceeding the capacity threshold
    trips_ex_capacity = [
        k for k in enschede[enschede['exceeding_capacity'] > 0]['TripNumber']]
    toAssign = enschede[enschede['TripNumber'].isin(trips_ex_capacity)]
    toAssign = toAssign['TripNumber'].drop_duplicates(keep='first').tolist()

    # list of trips that could potentially be re-assigned
    reAssign = enschede[~enschede['TripNumber'].isin(trips_ex_capacity)]
    reAssign = reAssign['TripNumber'].drop_duplicates(keep='first').tolist()

    # list of three following trips for each trip operated by the same bus
    sorted_data = data.sort_values(
        by=['IdVehicle', 'DepartureTime']).drop_duplicates(subset=['TripNumber'], keep='first')

    bus_trips.update({k: list(v)
                      for k, v in sorted_data.groupby('IdVehicle')['TripNumber']})

    all_following_trips = {}
    for trip in reAssign:
        list_ftrips = []
        for key, value in bus_trips.items():
            if trip in value:
                all_following_trips.update({trip: value})

    dep_time = data.sort_values(by=['Systeemlijnnr', 'DepartureTime']).drop_duplicates(
        subset=['TripNumber'], keep='first')
    dep_time_dict = {k: v for k, v in zip(
        dep_time['TripNumber'], dep_time['DepartureTime'])}

    for key, value in all_following_trips.items():
        following_trips_list = []
        for i in value:
            if dep_time_dict[key] < dep_time_dict[i] and i != key:
                following_trips_list += [i]
        all_following_trips.update({key: following_trips_list[0:4]})
    # remove trips which has no following trips
    for key, value in all_following_trips.items():
        if len(value) >= 2:
            trip_following_trips.update({key: value[0:4]})
    # precondition 1: a trip cannot be reassigned if the following trips running by the same bus is expected to be overcrowded
    for key, value in trip_following_trips.items():
        list_following_trips = []
        for i in value:
            if i not in toAssign:
                list_following_trips += [i]
            trip_following_trips.update({key: list_following_trips})
    # update the reassign list
    reAssign = [x for x in trip_following_trips.keys()]

    # list of preceeding trips on the same bus line
    sorted_data2 = data.sort_values(
        by=['Systeemlijnnr', 'DepartureTime']).drop_duplicates(subset=['TripNumber'], keep='first')

    line_trips.update({k: list(v)
                       for k, v in sorted_data2.groupby(['Systeemlijnnr', 'Direction'])['TripNumber']})

    all_preceeding_trips = {}
    for trip in toAssign:
        for key, value in line_trips.items():
            if trip in value:
                all_preceeding_trips.update({trip: value})
    preceeding_trip = {}
    for key, value in all_preceeding_trips.items():
        preceeding_list = []
        for i in value:
            if dep_time_dict[i] < dep_time_dict[key]:
                preceeding_list += [i]
                preceeding_list.reverse()
        preceeding_trip.update({key: preceeding_list[0]})

    return demand_dict, stranded_passenger_dict, stops_dict, toAssign, reAssign, trip_following_trips, preceeding_trip


'''
For creating list B, these preconditions should met:
   1. trips should depart before depature time of trips in A
   2. their following trips should not be overcrowded
'''


def bus_reassginment(data):
    time_window = 60000
    deadhead_dict = deadhead()
    demand_dict, stranded_passenger_dict, stops_dict, toAssign, reAssign, trip_following_trips, preceeding_trip = sets_parameters(
        data)

    waiting_time_dict = dp.waiting_time(data)

    first_stop_dict, last_stop_dict, dep_time_dict, arr_time_dict, travel_time_dict = dp.data_preprocessing(
        data)

    model = gp.Model('Bus Reassignment')
    epsilon = 120000  # this is the time for boarding passengers

    bigM = 1.64456466e+12  # big value
    deadhead_threshold = 900000  # deadhead time threshold is set 15 minutes
    # create pair of potential trips for re-assignemnt and trips that they could be re-assigned before
    if len(toAssign) > 0:
        pass
    else:
        print(colored("Error! No overcrowded trip was found",
              'green', attrs=['bold']))
        sys.exit()

    paired_trips = tuplelist()
    for i in reAssign:
        for j in toAssign:
            if dep_time_dict[i] + deadhead_dict[(first_stop_dict[i], first_stop_dict[j])] <= dep_time_dict[j] + time_window:
                paired_trips += [(i, j)]
    # x = model.addVars(trip_pairs, vtype=GRB.BINARY, name='x')
    reassign_var = model.addVars(
        paired_trips, vtype=GRB.BINARY, name="x[%s, %s]" % (i, j))
    # create pair of potential imposedcancellations
    imposed_cancellation = tuplelist()
    for i, j in paired_trips:
        for k in trip_following_trips[i]:
            imposed_cancellation += [(i, j, k)]

    imposed_var = model.addVars(
        imposed_cancellation, vtype=GRB.BINARY, name="x[%s, %s, %s]" % (i, j, k))
    model.update()
    # objective
    obj = quicksum(0.5 * reassign_var[i, j] * stranded_passenger_dict[j, s] * waiting_time_dict[j] for i, j in paired_trips for s in stops_dict[j]) + quicksum(3 * (1-reassign_var[i, j]) * stranded_passenger_dict[j, s] * waiting_time_dict[j] for i, j in paired_trips for s in stops_dict[j]) + quicksum(2 * reassign_var[i, j] *
                                                                                                                                                                                                                                                                                                             demand_dict[i, s] * waiting_time_dict[i] for i, j in paired_trips for s in stops_dict[i]) + quicksum(2 * imposed_var[i, j, k] * demand_dict[k, s] * waiting_time_dict[k] for i, j in paired_trips for k in trip_following_trips[i] for s in stops_dict[k])  # + quicksum(0.001 * imposed_var[i,j,k] for i, j, k in imposed_cancellation)
    model.setObjective(obj, GRB.MINIMIZE)
    model.update()
    # add constraints
    # assign only one trip
    model.addConstrs((reassign_var.sum('*', j) <=
                     1 for j in toAssign), name="toAssign[%s]" % j)
    # cancell a trip only once
    model.addConstrs((reassign_var.sum(i, '*') <=
                     1 for i in reAssign), name='cancellation[%s]' % i)
    # deadhead time should not exceed the threshold
    model.addConstrs((reassign_var[i, j] * deadhead_dict[(first_stop_dict[i], first_stop_dict[j])]
                     <= deadhead_threshold for i, j in paired_trips), name='deadhead[%s, %s]' % (i, j))
    # lastest departure time of re-assigned trips
    model.addConstrs((reassign_var[i, j] * (dep_time_dict[i] + deadhead_dict[(first_stop_dict[i], first_stop_dict[j])])
                     <= dep_time_dict[j] + time_window for i, j in paired_trips), name='departureTimeUp[%s, %s]' % (i, j))
    # model.addConstrs((reassign_var[i, j] * (dep_time_dict[i] + deadhead_dict[(first_stop_dict[i], first_stop_dict[j])])
    #                  >= dep_time_dict[j] - time_window for i, j in paired_trips), name='departureTimeUp[%s, %s]' % (i, j))
    # earliest departure time of re-assigned trips: before departure of the very first preceeding trip
    model.addConstrs((reassign_var[i, j] * (dep_time_dict[preceeding_trip[j]]) <= dep_time_dict[i] + deadhead_dict[(first_stop_dict[i], first_stop_dict[j])]
                      for i, j in paired_trips), name='departureTimeUp[%s, %s]' % (i, j))
    model.addConstrs((- reassign_var[i, j] + imposed_var[i, j, k] <=
                     0 for i, j in paired_trips for k in trip_following_trips[i]))
    # imposed cancellation
    for i, j in paired_trips:
        for p in trip_following_trips[i]:
            model.addConstr((reassign_var[i, j] * (dep_time_dict[i] + deadhead_dict[(first_stop_dict[i], first_stop_dict[j])] + travel_time_dict[j] + deadhead_dict[(
                last_stop_dict[j], first_stop_dict[p])])) - bigM * imposed_var[i, j, p] <= dep_time_dict[p])

    for i, j in paired_trips:
        # max number of imposed cancellation
        model.addConstrs((imposed_var.sum(i, '*', k) <=
                         2 for k in trip_following_trips[i]))
    model.update()

    model.optimize()
    model.__data1 = reassign_var
    model.__data2 = imposed_var
    active_arcs = [a for a in paired_trips if model.__data1[a].x > 0.99]
    print(colored("Optimal solutions:", 'green', attrs=['bold']))
    for i in active_arcs:
        print(
            "Trip {} can be re-assigned".format(i[0]) + ' ' + "before trip {}".format(i[1]))
    for j in toAssign:
        if j not in [a[1] for a in active_arcs]:
            print("No optimal bus trip was found to re-assign before trip {}".format(j))
    print(colored("Imposed Cancellations:", 'green', attrs=['bold']))
    imposed_arc = [
        a for a in imposed_cancellation if model.__data2[a].x > 0.99]
    for i in imposed_arc:
        print(
            "Trip {}".format(i[2]) + ' ' + " is cancelled as a result of cancelling trip {}".format(i[0]) + ' ' + 'and re-assigned before trip {}'.format(i[1]))
    for i in active_arcs:
        if i[0] not in [a[0] for a in imposed_arc]:
            print(
                "No imposed cancellation(s) due to re-assigning trip {}".format(i[0]) + ' ' + 'before trip {}'.format(i[1]))
    return model, active_arcs, imposed_arc


model, active_arcs, imposed_arc = bus_reassginment(dp.import_data(date))


"""
plot occupancy data
"""

# # print optimal solutions


def conv_millsecond_localtime(milliseconds):
    uct_time = dt.datetime.fromtimestamp(milliseconds/1000.0)
    uct_time = pytz.utc.localize(uct_time)
    cet = pytz.timezone('CET')
    offset = uct_time.astimezone(cet).utcoffset()
    local_time = uct_time - offset
    return local_time


def plot_occupancy(cancelled_trip, reassigned_trip):
    # data['DepartureTime'] = data['DepartureTime'].apply(
    #     conv_millsecond_localtime)
    line_number = data.sort_values(by=['Systeemlijnnr']).drop_duplicates(
        subset=['TripNumber'], keep='first')
    line_number = {i: j for i, j in zip(
        line_number.TripNumber, line_number.Systeemlijnnr)}
    cancelled = data[data['TripNumber'] == cancelled_trip]
    reassigned = data[data['TripNumber'] == reassigned_trip]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(cancelled.ActualDepartureTime,
             cancelled.OccupancyCorrected, color='green')
    ax1.legend(['Cancelled Trip'], loc=1)
    ax1.set(xlabel='Departure Time', ylabel='Occupancy')
    ax1.title.set_text('Line number: {} \n Trip number: {}'.format(
        line_number[cancelled_trip], cancelled_trip))
    ax1.tick_params(axis="x", rotation=45)

    ax2.plot(reassigned.ActualDepartureTime,
             reassigned.OccupancyCorrected, color='blue')
    ax2.legend(['Overcrowded Trip'], loc=1)
    ax2.set(xlabel='Departure Time', ylabel='Occupancy')
    ax2.title.set_text('Line number: {} \n Trip number: {}'.format(
        line_number[reassigned_trip], reassigned_trip))
    ax2.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    plt.savefig('Occupancy for trip {} assigned to {}.png'.format(
        cancelled_trip, reassigned_trip), bbox_inches='tight', dpi=300)
    plt.show()


for i, j in active_arcs:
    plot_occupancy(i, j)


# detailes of cancelled and re-assigned trips
def model_output(active_arcs, imposed_arc, data):
    if len(active_arcs) > 0:
        pass
    else:
        sys.exit()
    epsilon = 120000
    cursor, conn = dp.connect_to_database()
    bus_stops = pd.read_sql_query('select * from bus_stops', conn)
    bus_stops = {i: j for i, j in zip(
        bus_stops['IdDimHalte'], bus_stops['Naam_halte'])}
    cursor.close()

    deadhead_dict = deadhead()
    demand_dict, stranded_passenger_dict, stops_dict, toAssign, reAssign, trip_following_trips, preceeding_trip = sets_parameters(
        data)

    first_stop_dict, last_stop_dict, dep_time_dict, arr_time_dict, travel_time_dict = dp.data_preprocessing(
        data)
    # data['DepartureTime'] = data['DepartureTime'].apply(
    #     conv_millsecond_localtime)

    line_number = data.sort_values(by=['Systeemlijnnr']).drop_duplicates(
        subset=['TripNumber'], keep='first')
    line_number = {i: j for i, j in zip(
        line_number.TripNumber, line_number.Systeemlijnnr)}

    for i, j in active_arcs:
        cancel_linenr = line_number[i]
        cancel_departure = conv_millsecond_localtime(dep_time_dict[40869])
        cancel_first_stop = bus_stops[first_stop_dict[i]]
        reassign_linenr = line_number[j]
        toassign_departure = conv_millsecond_localtime(dep_time_dict[j])
        deadhead_time = (
            deadhead_dict[(first_stop_dict[i], first_stop_dict[j])])/60000
        reassigned_departure = conv_millsecond_localtime(
            dep_time_dict[i] + deadhead_dict[(first_stop_dict[i], first_stop_dict[j])] + epsilon)
        reassigned_first_stop = bus_stops[first_stop_dict[j]]

        for p, q, r in imposed_arc:
            if len(imposed_arc) > 0 and p == i and q == j:
                next_planned_trip = trip_following_trips[r][0]
                return_time = conv_millsecond_localtime(dep_time_dict[p] + deadhead_dict[(first_stop_dict[p], first_stop_dict[q])] +
                                                        epsilon + travel_time_dict[q] + deadhead_dict[(last_stop_dict[q], first_stop_dict[next_planned_trip])])
                ext_planned_linenr = line_number[next_planned_trip]
                next_planned_trip_departure = conv_millsecond_localtime(
                    dep_time_dict[next_planned_trip])
                next_trip_first_stop = bus_stops[first_stop_dict[r]]
            else:
                sys.exit()
        next_planned_trip = trip_following_trips[i][0]
        return_time = conv_millsecond_localtime(dep_time_dict[i] + deadhead_dict[(first_stop_dict[i], first_stop_dict[j])] +
                                                epsilon + travel_time_dict[j] + deadhead_dict[(last_stop_dict[j], first_stop_dict[next_planned_trip])])
        next_planned_trip_departure = conv_millsecond_localtime(
            dep_time_dict[next_planned_trip])
        next_planned_linenr = line_number[next_planned_trip]
        next_trip_first_stop = bus_stops[first_stop_dict[next_planned_trip]]

        print(colored('Optimal solution:', 'green', attrs=['bold']))
        print("Cancelled trip {}".format(i) + ' ' +
              'from line {}'.format(cancel_linenr))
        print("Assign before/after trip {}".format(j) +
              ' ' + 'on line {}'.format(reassign_linenr))
        print(colored('Cancellation details:', 'green', attrs=['bold']))
        print("Planned departure from: {}".format(
            cancel_first_stop) + ': {}'.format(cancel_departure))
        print("Deadhead time from: {}".format(cancel_first_stop) +
              " to: {}".format(reassigned_first_stop) + ": {}".format(deadhead_time) + ' minutes')
        print('Boarding time: 2 minutes')
        print("Re-assigned departure from {}".format(reassigned_first_stop) +
              ': {}'.format(reassigned_departure))
        print("Return to the next planned trip {}".format(
            next_planned_trip) + " " + "on line {}".format(next_planned_linenr) + ': {}'.format(return_time))
        print("Planned departure of {}".format(next_planned_trip) + ' ' +
              'from {}'.format(next_trip_first_stop) + ': {}'.format(next_planned_trip_departure))

        cursor, conn = dp.connect_to_database()
        model_output = pd.read_sql_query('select * from model_output', conn)
        if i not in list(model_output['cancelledTrip']):
            cursor.execute("INSERT INTO model_output (cancelledTrip, lineNumber, plannedDeparture, plannedStop, reassignTo, reassignDeparture, reassignStop, deadheadTime, boardingtime, reassignedDeparture, returnTime, nextTrip, nextLineNumber, nextDeparture, nextStop) values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           i, cancel_linenr, cancel_departure, cancel_first_stop, j, toassign_departure, reassigned_first_stop, deadhead_time, epsilon/60000, reassigned_departure, return_time, next_planned_trip, next_planned_linenr, next_planned_trip_departure, next_trip_first_stop)

            conn.commit()
        # cursor.execute("SELECT * FROM model_output GROUP BY cancelledTrip HAVING COUNT(cancelledTrip) > 1")
        cursor.close()


model_output(active_arcs, imposed_arc, data)


# # list of cancelled trips
# cancelled = [a[0] for a in active_arcs]
# reassigned_before = [a[1] for a in active_arcs]


# stranded_pas = quicksum(demand_dict[i, s]
#                         for i in cancelled for s in stops_dict[i])

# model.computeIIS()
# model.write("model.ilp")
