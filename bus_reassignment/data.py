import pandas as pd 
import numpy as np 
import datetime as dt
data = pd.read_csv(r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/bezig_new/bezig.csv',
                    sep=';')

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
enschede_data.drop(['IdDimConcessie', 'IdDimModaliteit', 'Bron', 'RitVertrekTijdInt', 'PasseertijdInt', 'HalteVolgNr', 'IdDimOVChipHalte', 'Code_halte', 'Lengtegraad', 'Breedtegraad'], axis=1, inplace=True)

enschede_data.to_csv(r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/data_enschede.csv', sep=";")

# correct the data type

# select data only for one day 
test_data = enschede_data[enschede_data['IdDimDatum'] <= 20220211]

bus_lines = pd.read_csv(r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/Twente_bus_lines.csv',
                    sep=';')

bus_stops = pd.read_csv(r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/Twente_bus_stops.csv',
                    sep=';')

# merge data with line direction 
merged_data = pd.merge(data, Line_direction, on=["IdDimLijnRichting", "IdDimLijnRichting"])
# merge data with bus stops
merged_data = pd.merge(merged_data, bus_stops, on=["IdDimHalte", "IdDimHalte"])

# keep only necessary columns
merged_data = pd.DataFrame(merged_data[["IdDimDatum", "Ritnummer", "IdDimTijdBlok", "PublieksLijnnr", "RitVertrekTijd", "Passeertijd", "RitVertrekTijdInt", "IdDimHalte", "Naam_halte", "Bezetting"]])

trip = merged_data[merged_data["Ritnummer"]==40708]

# export data to csv file
merged_data.to_csv(r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/Trip_numbers.csv')
