"""
Created on Mon Jan 11 11:09:05 2021

@author: FarahmandZH
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta, datetime
import pyodbc


def connect_to_database():
    conn = pyodbc.connect('Driver={SQL Server Native Client 11.0};'
                          'Server=_insert server name_;'
                          'Database=_database name_;'
                          'Trusted_Connection=yes;')
    cursor = conn.cursor()
    return cursor, conn

def import_data():
    # call data from the database
    cursor, conn = connect_to_database()
    data = pd.read_sql_query(
        "select * from pred_occupancy_per_stop where operating_date = '{}' ".format(date), conn)
    cursor.close()

data = import_data()

#  find null values in the data and fill out or drop the nulls
print(data.isnull().sum())

# check the data type if every column is imported correctly
print(data.dtypes)


"""
Make sure that the data type and format are correct. 
"""
# columns names are in Dutch, so change them to English
# data.rename(columns={'IdFactRit': 'Id', 'IdDimDatum': 'Date', 'Concessiecode': 'Doncession_code', 'PublieksLijnnr': 'Public_linenr', 'Systeemlijnnr': 'System_linenr', 'Naam_lijn': 'Line_name'}, inplace=True)


"""
Bus services (in our case study) have time-blocks from 4 to 27. Therefore, the time block 24, 25, 26, and 27 should be shifted to the next day since the system in Keolis records them as the same day as previous time blocks. 
"""
data['adjusted_date'] = data.apply(
    lambda row: row['date'] + timedelta(1) if row['Stepin_timeblock']>=24 else row['date'],
    axis=1)
data['adjusted_date'] = pd.to_datetime(data['adjusted_date']).dt.date
# Date is no longer needed, so remove and use adjusted column instead
data = data.drop('date', axis=1)

# Create a column for converting timeblock to normal time
data['Time'] = ''

#%%
# convert time blocks to normal hours (4 == 4:00:00)
data['Time'] = np.where(data['Stepin_timeblock'] ==4, '4:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==5, '5:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==6, '6:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==7, '7:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==8, '8:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==9, '9:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==10, '10:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==11, '11:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==12, '12:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==13, '13:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==14, '14:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==15, '15:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==16, '16:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==17, '17:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==18, '18:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==19, '19:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==20, '20:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==21, '21:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==22, '22:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==23, '23:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==24, '00:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==25, '1:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==26, '2:00:00', data['Time'])
data['Time'] = np.where(data['Stepin_timeblock'] ==27, '3:00:00', data['Time'])

# combine date and time columns into one colums so to make a single key for combination with weather data
data['DateTime'] = data['adjusted_date'].map(str) + ' ' + data['Time'].map(str) 
print(data.head(10))
# change name of adjusted data to date
data.rename(columns={'adjusted_date': 'Date'}, inplace=True)

# convert data type to datetime format 
data['DateTime'] =  pd.to_datetime(data['DateTime'], infer_datetime_format=True)
data['Date'] = pd.to_datetime(data['Date']).dt.date

# Group data per datetime and line number 
"""
SystemLijnnr = Bus line number
IdFactRit = Trip Id
"""
grouped_data = pd.DataFrame(data.groupby(['DateTime', 'Systeemlijnnr'], as_index=False)['IdFactRit'].count().rename(columns={'IdFactRit': 'Total_trips'}))

grouped_data['DateTime'] =  pd.to_datetime(grouped_data['DateTime'], infer_datetime_format=True)
# grouped_data['Date'] = pd.to_datetime(grouped_data['DateTime']).dt.date
grouped_data = grouped_data.sort_values('DateTime')

# Export cleaned data 



















