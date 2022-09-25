import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import pyodbc


def connect_to_database():
    conn = pyodbc.connect('Driver={SQL Server Native Client 11.0};'
                          'Server=ZAKIR;'
                          'Database=keolis;'
                          'Trusted_Connection=yes;')
    cursor = conn.cursor()
    return cursor, conn

def connect_to_databaseapi():
    conn = pyodbc.connect('Driver={SQL Server Native Client 11.0};'
                          'Server=ZAKIR;'
                          'Database=keodss3.0;'
                          'Trusted_Connection=yes;')
    cursor = conn.cursor()
    return cursor, conn


def export_occupancy_per_trip(data_name):
    data = pd.read_csv(
        r'C:/Users/zfara/OneDrive - University of Twente/Documenten/PDEng Project/Data/{}.csv'.format(data_name), sep=';')
    data = data.fillna(0)

    cursor, conn = connect_to_database()
    column = ['OperatingDate', 'TripNumber', 'ActualDepartureTime', 'Systeemlijnnr', 'Direction', 'IdDimHalte',
              'OccupancyCorrected'
              ]
    data['ActualDepartureTime'] = pd.to_datetime(
        data['ActualDepartureTime'], infer_datetime_format=True)
    data = data[column]
    data = data.values.tolist()
    sql_insert = '''
        declare @operating_date date = ?
        declare @trip_number bigint = ?
        declare @departure_time datetime = ?
        declare @system_linenr bigint = ?
        declare @direction bigint = ?
        declare @stop bigint = ?

        declare @occupancy float = ?

        UPDATE occupancy_per_trip  
        SET departure_time = @departure_time, occupancy=@occupancy
        WHERE operating_date = @operating_date AND trip_number = @trip_number AND system_linenr =@system_linenr AND direction = @direction AND stop=@stop

        IF @@ROWCOUNT = 0
            INSERT INTO occupancy_per_trip
                (operating_date, trip_number, departure_time, system_linenr, direction, stop, occupancy)
            VALUES (@operating_date, @trip_number, @departure_time, @system_linenr, @direction, @stop, @occupancy)
        '''
    cursor.executemany(sql_insert, data)

    conn.commit()
    cursor.close()


export_occupancy_per_trip(data_name='September1_7')


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

    cursor, conn = connect_to_database()
    data = np.array(data)
    for i in range(len(data)):
        cursor.execute("INSERT INTO bus_cancellations (date_time, system_linenr, direction, num_cancellations) values(?,?,?,?)",
                       data[i, 0], data[i, 1], data[i, 2], data[i, 3])

    conn.commit()
    cursor.close()


export_cancellation(data_name='bus_cancellation')


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

    cursor, conn = connect_to_database()
    data = np.array(data)
    for i in range(len(data)):
        cursor.execute("INSERT INTO boarding_data (date_time, system_linenr, direction, stop, boarders) values(?,?,?,?,?)",
                       data[i, 0], data[i, 1], data[i, 2], data[i, 3], data[i, 4])

    conn.commit()
    cursor.close()


export_boarding(data_name='boarding_alighting_2021')


def export_occupancy_per_stop(data_name):
    # data = pd.read_csv(
    #     r'C:/Users/FarahmandZH/OneDrive - University of Twente/Documenten/PDEng Project/Data/{}.csv'.format(data_name), sep=';')
    data = pd.read_csv(
        r'C:/Users/zfara/OneDrive - University of Twente/Documenten/PDEng Project/Data/{}.csv'.format(data_name), sep=';')
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
    cursor, conn = connect_to_database()
    data = np.array(data)
    for i in range(len(data)):
        cursor.execute("INSERT INTO occupancy_per_stop (date_time, system_linenr, direction, stop, occupancy) values(?,?,?,?,?)",
                       data[i, 0], data[i, 1], data[i, 2], data[i, 3], data[i, 4])

    conn.commit()
    cursor.close()


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
    cursor, conn = connect_to_database()
    data = np.array(data)
    for i in range(len(data)):
        cursor.execute("INSERT INTO weather_data (date_time, temp, sun_duration, prec_duration, prec_amount, cloud_cover, humidity, rain, snow, windx, windy) values(?,?,?,?,?,?,?,?,?,?,?)",
                       data[i, 0], data[i, 1], data[i, 2], data[i, 3], data[i, 4], data[i, 5], data[i, 6], data[i, 7], data[i, 8], data[i, 9], data[i, 10])

    conn.commit()
    cursor.close()


export_weather(data_name='weather_2011_2020')


def calculate_deadhead(data):
    data = pd.read_csv(
        r'C:/Users/zfara/OneDrive - University of Twente/Documenten/PDEng Project/Data/deadhead_time.csv', sep=',')
    cursor, conn = connect_to_database()
    data = data.values.tolist()
    sql_insert = '''
        declare @stopA bigint = ?
        declare @stopB bigint = ?
        declare @deadhead float = ?

        UPDATE deadhead_time    
        SET deadhead = @deadhead
        WHERE stopA = @stopA AND stopB = @stopB

        IF @@ROWCOUNT = 0
            INSERT INTO deadhead_time
                (stopA, stopB, deadhead)
            VALUES (@stopA, @stopB, @deadhead)
        '''
    cursor.executemany(sql_insert, data)

    conn.commit()
    cursor.close()


# def export_timetable():
#     cursor, conn = connect_to_database()

#     data = pd.read_sql("SELECT * FROM timetable_data", conn)
#     data.dropna(inplace=True)
#     cursor.close()
#     data.sort_values(
#         by=['IdDimDatum', 'Systeemlijnnr', 'dep_time'], inplace=True)
#     column = ['IdDimDatum', 'Systeemlijnnr', 'Richting', 'RitNummer', 'Volgnummer',
#               'IdDimHalte', 'passing_time', 'arr_time', 'dep_time']
#     data = data[column]

#     data = data.values.tolist()
#     conn = pyodbc.connect('Driver={SQL Server Native Client 11.0};'
#                           'Server=ZAKIR;'
#                           'Database=keodss;'
#                           'Trusted_Connection=yes;')
#     cursor = conn.cursor()
#     sql_insert = '''
#         declare @operating_date date = ?
#         declare @system_linenr bigint = ?
#         declare @direction bigint = ?
#         declare @trip_number bigint = ?
#         declare @vehicle_number bigint = ?
#         declare @stop bigint = ?
#         declare @passing_time datetime = ?
#         declare @arrival_time datetime = ?
#         declare @departure_time datetime = ?


#         INSERT INTO api_timetable
#                 (operating_date, system_linenr, direction, trip_number, vehicle_number, stop, passing_time, arrival_time, departure_time)
#             VALUES (@operating_date, @system_linenr, @direction, @trip_number, @vehicle_number, @stop, @passing_time, @arrival_time, @departure_time)
#         '''
#     cursor.executemany(sql_insert, data)

#     conn.commit()
#     cursor.close()

def export_timetable_to_api():
    cursor, conn = connect_to_database()

    data = pd.read_csv(
        r'C:/Users/zfara/OneDrive - University of Twente/Documenten/PDEng Project/Data/timetable.csv', sep=';')
    stops = pd.read_sql("SELECT * FROM bus_stops", conn)

    data = pd.merge(data, stops, on=['IdDimHalte', 'IdDimHalte'])

    """Note: some trips on line 2 start from the central station instead of Disselhoek """

    data = data.sort_values(by=['IdDimDatum', 'RitNummer', 'passing_time']).drop_duplicates(
        subset=['IdDimDatum', 'RitNummer', 'Systeemlijnnr', 'Richting'], keep='first')

    data['IdDimDatum'] = pd.to_datetime(
        data['IdDimDatum'].astype(str), format='%Y%m%d')
    data['IdDimDatum'] = data['IdDimDatum'].astype(str)
    
    data['date_time'] = data['IdDimDatum'] + ' ' + data['dep_time']
    data['date_time'] = pd.to_datetime(data['date_time'], infer_datetime_format=True)
    
    data = data.sort_values(by=['Systeemlijnnr', 'Richting', 'RitNummer', 'date_time'])
    data = data.reset_index()
    column = ['IdDimDatum', 'Systeemlijnnr', 'Richting', 'RitNummer', 'IdDimHalte', 'Naam_halte',
              'passing_time', 'dep_time', 'date_time']
    data = data[column]
    data = data.values.tolist()
    cursor, conn = connect_to_databaseapi()
    sql_insert = '''
        declare @operating_date date = ?
        declare @system_linenr bigint = ?
        declare @direction bigint = ?
        declare @trip_number bigint = ?
        declare @stop bigint = ?
        declare @stop_name nvarchar(50) = ?
        declare @passing_time time = ?
        declare @departure_time time = ?
        declare @date_time datetime = ?

        
        INSERT INTO api_timetable
                (operating_date, system_linenr, direction, trip_number, stop, stop_name, passing_time, departure_time, date_time)
            VALUES (@operating_date, @system_linenr, @direction, @trip_number, @stop, @stop_name, @passing_time, @departure_time, @date_time)
        '''
    cursor.executemany(sql_insert, data)

    conn.commit()
    cursor.close()




def export_trips_timetable(date):
    cursor, conn = connect_to_database()

    data = pd.read_sql(
        "SELECT * FROM timetable WHERE operating_date = '{}'".format(date), conn)
    data = data.replace({np.nan: None})

    for i in range(len(data)):
        if data.loc[i, 'passing_time'] is None:
            data.loc[i, 'passing_time'] = data.loc[i, 'arrival_time']
        else:
            data.loc[i, 'passing_time'] = data.loc[i, 'passing_time']

    """Note: some trips on line 2 start from the central station instead of Disselhoek """
    first_stops = data.sort_values(by=['operating_date', 'trip_number', 'passing_time']).drop_duplicates(
        subset=['operating_date', 'trip_number', 'system_linenr', 'direction'], keep='first')

    last_stops = data.sort_values(by=['operating_date', 'trip_number', 'passing_time']).drop_duplicates(
        subset=['operating_date', 'trip_number', 'system_linenr', 'direction'], keep='last')

    data = pd.merge(first_stops, last_stops, on=[
                    'operating_date', 'trip_number'])

    column = ['operating_date', 'system_linenr_x', 'direction_x', 'trip_number',
              'stop_x', 'departure_time_x', 'stop_y', 'passing_time_y',
              'vehicle_number_x']
    data = data[column]
    data = data.astype(object).where(pd.notnull(data), None)
    data = data.values.tolist()

    sql_insert = '''
        declare @operating_date date = ?
        declare @system_linenr bigint = ?
        declare @direction bigint = ?
        declare @trip_number bigint = ?
        declare @start_stop bigint = ?
        declare @departure_time time = ?
        declare @last_stop bigint= ?
        declare @arrival_time time = ?
        declare @vehicle_number bigint = ?

        INSERT INTO trips_timetable
                (operating_date, system_linenr, direction, trip_number, start_stop, departure_time, last_stop, arrival_time, vehicle_number)
            VALUES (@operating_date, @system_linenr, @direction, @trip_number, @start_stop, @departure_time, @last_stop, @arrival_time, @vehicle_number)
        '''
    cursor.executemany(sql_insert, data)

    conn.commit()
    cursor.close()


def export_stops():

    cursor, conn = connect_to_database()
    data = pd.read_sql_query(
        'select * from timetable', conn)
    stops = pd.read_sql("Select * from bus_stops", conn)
    cursor.close()

    lines_list = [4701, 4702, 4703, 4704,
                  4705, 4706, 4707, 4708, 4709, 4060, 4061, 4062]
    data = data[data['system_linenr'].isin(lines_list)]

    
    
    data = data.drop_duplicates(
        subset=['system_linenr', 'direction', 'stop'], keep='first')
    
    data = pd.merge(data, stops, left_on=['stop'], right_on=['IdDimHalte'])
    data = data.drop_duplicates(
        subset=['system_linenr', 'direction', 'stop'], keep='first')
    
    conn = pyodbc.connect('Driver={SQL Server Native Client 11.0};'
                          'Server=ZAKIR;'
                          'Database=keodss2.0;'
                          'Trusted_Connection=yes;')
    cursor = conn.cursor()
    data = data[['system_linenr', 'direction', 'stop', 'Naam_halte']]
    
    data = data.values.tolist()
    sql_insert = '''
        declare @system_linenr bigint = ?
        declare @direction bigint = ?
        declare @stop bigint = ?
        declare @stop_name nvarchar(50) = ?

        INSERT INTO api_busstops
            (system_linenr, direction, stop, stop_name)
        VALUES (@system_linenr, @direction, @stop, @stop_name)
        '''
    cursor.executemany(sql_insert, data)

    conn.commit()
    cursor.close()



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
