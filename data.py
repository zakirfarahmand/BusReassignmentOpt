import pandas as pd 
import numpy as np 

data = pd.read_csv(r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/bezig_new/bezig.csv',
                    sep=';')

bus_lines = pd.read_csv(r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/Twente_bus_lines.csv',
                    sep=';')

bus_stops = pd.read_csv(r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/Twente_bus_stops.csv',
                    sep=';')

Line_direction = pd.read_csv(r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/Line_direction.csv',
                    sep=';')

data = data[data['IdDimDatum']==20220302]


conc = pd.merge(data, Line_direction, on=["IdDimLijnRichting", "IdDimLijnRichting"], how="left")

merged_data = pd.merge(conc, bus_lines, on=["Systeemlijnnr", 'Systeemlijnnr'], how='left')

merged_data.to_csv(r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/Trip_number.csv')
