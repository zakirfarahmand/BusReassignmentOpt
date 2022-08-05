import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import data_preprocessing as dp
import datetime as dt


def export_occupancy_per_trip(data_name):
    data = pd.read_csv(
        r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/Occupancy2022/{}.csv'.format(data_name), sep=';')
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


export_occupancy_per_trip(data_name='boarding_alighting_2021')


def export_cancellation(data_name):
    data = pd.read_csv(
        r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/{}.csv'.format(data_name), sep=',')

    data['Date'] = pd.to_datetime(
        data['IdDimDatum'].astype(str), format='%Y%m%d')
    data['Time'] = ''
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            4, '4:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            5, '5:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            6, '6:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            7, '7:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            8, '8:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            9, '9:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            10, '10:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            11, '11:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            12, '12:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            13, '13:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            14, '14:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            15, '15:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            16, '16:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            17, '17:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            18, '18:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            19, '19:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            20, '20:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            21, '21:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            22, '22:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            23, '23:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            24, '00:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            25, '1:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            26, '2:00:00', data['Time'])
    data['Time'] = np.where(data['IdDimTijdBlok'] ==
                            27, '3:00:00', data['Time'])

    data['DateTime'] = data['Date'].map(str) + ' ' + data['Time'].map(str)
    data['DateTime'] = pd.to_datetime(
        data['DateTime'], infer_datetime_format=True)

    data = pd.DataFrame(data.groupby(['DateTime', 'Systeemlijnnr', 'Richting'])[
        'Concession'].count().reset_index())

    cursor, conn = dp.connect_to_database()
    data = np.array(data)
    for i in range(len(data)):
        cursor.execute("INSERT INTO bus_cancellations (date_time, system_linenr, direction, num_cancellations) values(?,?,?,?)",
                       data[i, 0], data[i, 1], data[i, 2], data[i, 3])

    conn.commit()
    cursor.close()


export_cancellation(data_name='bus_cancellation')


data_name = 'boarding_alighting_2022'


def export_boarding(data_name):
    data = pd.read_csv(
        r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/{}.csv'.format(data_name), sep=';')
    data = data.fillna(0)
    data['hour'] = pd.to_datetime(data['ActualDepartureTime']).dt.hour

    data = pd.DataFrame(data.groupby(['OperatingDate', 'hour', 'Systeemlijnnr', 'Direction', 'IdDimHalte'])[
                        'BoardersCorrected'].sum().reset_index())

    data['Time'] = ''

    data['Time'] = np.where(data['hour'] == 4, '4:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 5, '5:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 6, '6:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 7, '7:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 8, '8:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 9, '9:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 10, '10:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 11, '11:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 12, '12:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 13, '13:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 14, '14:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 15, '15:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 16, '16:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 17, '17:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 18, '18:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 19, '19:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 20, '20:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 21, '21:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 22, '22:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 23, '23:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 0, '00:00:00', data['Time'])
    data['DateTime'] = data['OperatingDate'].map(
        str) + ' ' + data['Time'].map(str)
    data['DateTime'] = pd.to_datetime(
        data['DateTime'], infer_datetime_format=True)

    column = ['DateTime', 'Systeemlijnnr',
              'Direction', 'IdDimHalte', 'BoardersCorrected']
    data = data[column]

    cursor, conn = dp.connect_to_database()
    data = np.array(data)
    for i in range(len(data)):
        cursor.execute("INSERT INTO boarding_data (date_time, system_linenr, direction, stop, boarders) values(?,?,?,?,?)",
                       data[i, 0], data[i, 1], data[i, 2], data[i, 3], data[i, 4])

    conn.commit()
    cursor.close()


export_boarding(data_name='boarding_alighting_2021')


def export_occupancy_per_stop(data_name):
    data = pd.read_csv(
        r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/{}.csv'.format(data_name), sep=';')
    data = data.fillna(0)
    data['hour'] = pd.to_datetime(data['ActualDepartureTime']).dt.hour

    data = pd.DataFrame(data.groupby(['OperatingDate', 'hour', 'Systeemlijnnr', 'Direction', 'IdDimHalte'])[
                        'OccupancyCorrected'].sum().reset_index())

    data['Time'] = ''

    data['Time'] = np.where(data['hour'] == 4, '4:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 5, '5:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 6, '6:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 7, '7:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 8, '8:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 9, '9:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 10, '10:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 11, '11:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 12, '12:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 13, '13:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 14, '14:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 15, '15:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 16, '16:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 17, '17:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 18, '18:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 19, '19:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 20, '20:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 21, '21:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 22, '22:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 23, '23:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 0, '00:00:00', data['Time'])
    data['DateTime'] = data['OperatingDate'].map(
        str) + ' ' + data['Time'].map(str)
    data['DateTime'] = pd.to_datetime(
        data['DateTime'], infer_datetime_format=True)

    column = ['DateTime', 'Systeemlijnnr',
              'Direction', 'IdDimHalte', 'OccupancyCorrected']
    data = data[column]
    cursor, conn = dp.connect_to_database()
    data = np.array(data)
    for i in range(len(data)):
        cursor.execute("INSERT INTO occupancy_data (date_time, system_linenr, direction, stop, occupancy) values(?,?,?,?,?)",
                       data[i, 0], data[i, 1], data[i, 2], data[i, 3], data[i, 4])

    conn.commit()
    cursor.close()


export_occupancy_per_stop(data_name='boarding_alighting_2021')
export_occupancy_per_stop(data_name='boarding_alighting_2022')


def export_weather(data_name):
    data = pd.read_csv(
        r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/{}.csv'.format(data_name), sep=',')
    column = ['YYYYMMDD', '   HH', '   DD', '   FF', '    T',
              '   SQ', '   DR', '   RH', '    N', '    U', '    R', '    S']
    data = data[column]
    data = data.rename(columns={'YYYYMMDD': 'Date', '   HH': 'hour', '   DD': 'wind_direction', '   FF': 'wind_speed', '    T': 'temp', '   SQ': 'sun_duration',
                       '   DR': 'prec_duration', '   RH': 'prec_amount', '    N': 'cloud_cover', '    U': 'humidity', '    R': 'rain', '    S': 'snow'})
    data = data.fillna(0)

    data['Time'] = ''
    data['Time'] = np.where(data['hour'] == 1, '1:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 2, '2:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 3, '3:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 4, '4:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 5, '5:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 6, '6:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 7, '7:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 8, '8:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 9, '9:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 10, '10:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 11, '11:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 12, '12:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 13, '13:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 14, '14:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 15, '15:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 16, '16:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 17, '17:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 18, '18:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 19, '19:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 20, '20:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 21, '21:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 22, '22:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 23, '23:00:00', data['Time'])
    data['Time'] = np.where(data['hour'] == 24, '00:00:00', data['Time'])
    data['DateTime'] = data['Date'].map(
        str) + ' ' + data['Time'].map(str)
    data['DateTime'] = pd.to_datetime(
        data['DateTime'], infer_datetime_format=True)
    data = data.sort_values(by=['DateTime']).drop_duplicates(
        'DateTime', keep='last')

    # convert to original values
    data['wind_speed'] = data['wind_speed']/10
    data['temp'] = data['temp']/10
    data['sun_duration'] = data['sun_duration']/10
    data['prec_duration'] = data['prec_duration']/10
    data['prec_amount'] = data['prec_amount']/10

    data['prec_amount'] = data['prec_amount'].apply(
        lambda x: 0.05 if x == -0.1 else x)
    # convert wind speed and direction to wind speed in x and y
    data['wv'] = data.pop('wind_speed')
    # Convert to radians.
    data['wd_rad'] = data.pop('wind_direction')*np.pi / 180
    # Calculate the wind x and y components.
    data['windx'] = data['wv']*np.cos(data['wd_rad'])
    data['windy'] = data['wv']*np.sin(data['wd_rad'])
    data['windx'] = round(data['windx'], 3)
    data['windy'] = round(data['windy'], 3)
    column = ['DateTime', 'temp', 'sun_duration', 'prec_duration', 'prec_amount',
              'cloud_cover', 'humidity', 'rain', 'snow', 'windx', 'windy']
    data = data[column]
    cursor, conn = dp.connect_to_database()
    data = np.array(data)
    for i in range(len(data)):
        cursor.execute("INSERT INTO weather_data (date_time, temp, sun_duration, prec_duration, prec_amount, cloud_cover, humidity, rain, snow, windx, windy) values(?,?,?,?,?,?,?,?,?,?,?)",
                       data[i, 0], data[i, 1], data[i, 2], data[i, 3], data[i, 4], data[i, 5], data[i, 6], data[i, 7], data[i, 8], data[i, 9], data[i, 10])

    conn.commit()
    cursor.close()


export_weather(data_name='weather_2011_2020')


# test = data[data['Systeemlijnnr'] == 4709]


# test1 = test[test['Direction'] == 1]
# test2 = test[test['Direction'] == 2]

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
# ax1.plot(test1.DateTime, test1.BoardersCorrected, color='green')
# ax1.legend(['direction 1'], loc=2)
# ax2.plot(test2.DateTime,
#          test2.BoardersCorrected, color='blue')
# # ax2.legend(['Planned departure'], loc=2)
# # ax2.set(xlabel='Departure Time', ylabel='Occupancy')
# # ax2.title.set_text('2022-02-11 \n trip number: {}'.format(trip_number))
# fig.tight_layout()
# plt.show()
