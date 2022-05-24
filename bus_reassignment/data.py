import pandas as pd 
import numpy as np 
import datetime as dt
from datetime import timedelta
data = pd.read_csv(r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/bezig_new/bezig.csv',
                    sep=';')

data['date'] = pd.to_datetime(data['IdDimDatum'].astype(str), format='%Y%m%d')

data['date'] = data.apply(
    lambda row: row['date'] + timedelta(1) if row['IdDimTijdBlok']>=24 else row['date'],
    axis=1)
# select only enschede lines 
enschede_lines = pd.read_csv(r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/Enschede_lines.csv',
                    sep=';') 

# enschede data 
enschede_data = pd.merge(data, enschede_lines, on=["IdDimLijnRichting", "IdDimLijnRichting"])

# bus stops 
bus_stops = pd.read_csv(r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/Twente_bus_stops.csv',
                    sep=';')
enschede_data = pd.merge(enschede_data, bus_stops, on=["IdDimHalte", "IdDimHalte"])

# remove unnecessary colums
enschede_data.drop(['IdFactBezetting', 'IdDimConcessie', 'IdDimModaliteit', 'Bron', 'RitVertrekTijdInt', 'PasseertijdInt', 'HalteVolgNr', 'IdDimOVChipHalte', 'Code_halte', 'Lengtegraad', 'Breedtegraad'], axis=1, inplace=True)


enschede_data.to_csv(r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/data_enschede.csv', sep=";")

