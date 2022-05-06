import pandas as pd 
import numpy as np 

data = pd.read_csv(r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/bezig_new/bezig.csv',
                    sep=';')
# select data only for one day 
data = data[data['IdDimDatum'] <= 20210302]

bus_lines = pd.read_csv(r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/Twente_bus_lines.csv',
                    sep=';')

bus_stops = pd.read_csv(r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/Twente_bus_stops.csv',
                    sep=';')

Line_direction = pd.read_csv(r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/Line_direction.csv',
                    sep=';')
Line_direction.drop(0, inplace=True)
enschede_lines = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
Line_direction = pd.DataFrame(Line_direction.loc[(Line_direction.PublieksLijnnr.isin(enschede_lines)),:]) # select only enschede bus lines

# merge data with line direction 
merged_data = pd.merge(data, Line_direction, on=["IdDimLijnRichting", "IdDimLijnRichting"])
# merge data with bus stops
merged_data = pd.merge(merged_data, bus_stops, on=["IdDimHalte", "IdDimHalte"])

# keep only necessary columns
merged_data = pd.DataFrame(merged_data[["IdDimDatum", "Ritnummer", "IdDimTijdBlok", "PublieksLijnnr", "RitVertrekTijd", "Passeertijd", "RitVertrekTijdInt", "IdDimHalte", "Naam_halte", "Bezetting"]])

trip = merged_data[merged_data["Ritnummer"]==40708]

# export data to csv file
merged_data.to_csv(r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/Trip_numbers.csv')
