import pandas as pd
import numpy as np
import data_preprocessing as dp

data_name = 'Occupancy2022'

data = pd.read_csv(
    r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/Occupancy2022/{}.csv'.format(data_name), sep=';')


def export_to_sql(data):
    data = data.fillna(0)

    data['Breedtegraad'] = data['Breedtegraad'].astype('float')
    data['Lengtegraad'] = data['Lengtegraad'].astype('float')
    data['TripNumber'] = data['TripNumber'].astype('int')
    data['Systeemlijnnr'] = data['Systeemlijnnr'].astype('int')
    data['IdVehicle'] = data['IdVehicle'].astype('int')
    data['IdDimHalte'] = data['IdDimHalte'].astype('int')
    data['OccupancyCorrected'] = data['OccupancyCorrected'].astype('int')
    cursor, conn = dp.connect_to_database()
    column = ['TripNumber', 'Direction', 'Systeemlijnnr', 'IdVehicle', 'OperatingDate',
              'ActualDepartureTime', 'DepartureTime', 'IdDimHalte', 'Breedtegraad', 'Lengtegraad', 'OccupancyCorrected']
    data = data[column]
    data = np.array(data)
    for i in range(len(data)):
        cursor.execute("INSERT INTO data (TripNumber, Direction, Systeemlijnnr, IdVehicle, OperatingDate, ActualDepartureTime, DepartureTime, IdDimHalte, Breedtegraad, Lengtegraad, OccupancyCorrected) values(?,?,?,?,?,?,?,?,?,?,?)",
                       data[i, 0], data[i, 1], data[i, 2], data[i, 3], data[i, 4], data[i, 5], data[i, 6], data[i, 7], data[i, 8], data[i, 9], data[i, 10])

    conn.commit()
    cursor.close()
