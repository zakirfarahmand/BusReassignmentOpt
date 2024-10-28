from datetime import datetime
import googlemaps
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta

# select only enschede lines
enschede_lines = pd.read_csv(r'[path]',
                             sep=';')

# enschede data
enschede_data = pd.merge(data, enschede_lines, on=[
                         "IdDimLijnRichting", "IdDimLijnRichting"])

# bus stops
bus_stops = pd.read_csv(r'[path]',
                        sep=';')


gmaps = googlemaps.Client(key='[key]')

now = datetime.now()

location = []
for i in len(bus_stops):
    location = [bus_stops['Lengtegraad'], bus_stops['Breedtegraad']]


directions_result = gmaps.directions("52.249605294727324, 6.855930421536205",
                                     "52.22221149666158, 6.889309051743431",
                                     mode="driving",
                                     avoid="ferries",
                                     departure_time=now
                                     )

print(directions_result[0]['legs'][0]['distance']['text'])
print(directions_result[0]['legs'][0]['duration']['text'])


enschede_data = pd.merge(enschede_data, bus_stops, on=[
                         "IdDimHalte", "IdDimHalte"])

# remove unnecessary colums
enschede_data.drop(['IdFactBezetting', 'IdDimConcessie', 'IdDimModaliteit', 'Bron', 'RitVertrekTijdInt', 'PasseertijdInt',
                   'HalteVolgNr', 'IdDimOVChipHalte', 'Code_halte', 'Lengtegraad', 'Breedtegraad'], axis=1, inplace=True)


# data.to_csv(r'[path]', sep=";")
