"""
Created on Mon Jan 11 11:09:05 2021

@author: FarahmandZH
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r'C:/Users/FarahmandZH/OneDrive - Universiteit Twente/Documenten/PDEng project/Data/2021 test/2021.csv', sep=';')

#  find null values in the data 
print(data.isnull().sum())

# since the number of null values is very small compare to the entire data, so remove rows with null values 
data = data.dropna()
#  check that no null values remain in the data
print(data.isnull().sum())

# check the data type if every column is imported correctly
print(data.dtypes)

# the data type for stepout timeblock should be integer, not float 
data['Stepin_timeblock'] = data['Stepin_timeblock'].astype('int64')

# change dutum format from integer to datetime 
data['date'] = pd.to_datetime(data['IdDimDatum'].astype(str), format='%Y%m%d')
# recheck the data type if everything is correct
print(data.dtypes)

"""
Now the data format is correct
"""
# columns names are in Dutch, so change them to English
# data.rename(columns={'IdFactRit': 'Id', 'IdDimDatum': 'Date', 'Concessiecode': 'Doncession_code', 'PublieksLijnnr': 'Public_linenr', 'Systeemlijnnr': 'System_linenr', 'Naam_lijn': 'Line_name'}, inplace=True)


"""
The time block 24, 25, 26, and 27 should be shifted to the next day since the system in Keolis records them as the same day as previous time blocks. 
"""
from datetime import timedelta, datetime
data['adjusted_date'] = data.apply(
    lambda row: row['date'] + timedelta(1) if row['Stepin_timeblock']>=24 else row['date'],
    axis=1)
data['adjusted_date'] = pd.to_datetime(data['adjusted_date']).dt.date
# Date is no longer needed, so remove and use adjusted column instead
data = data.drop('date', axis=1)

# Create a column for converting timeblock to normal time
data['Time'] = ''
#%%
"""
The question is weather time block 4 is equal to 4:00 or 5:00
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

"""
#%%
# # convert time blocks to normal hours (4 == 4:00:00)
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

descriptives = data.describe().transpose()
# combine date and time columns into one colums so to make a single key for combination with weather data
data['DateTime'] = data['adjusted_date'].map(str) + ' ' + data['Time'].map(str) 
print(data.head(10))
# change name of adjusted data to date
data.rename(columns={'adjusted_date': 'Date'}, inplace=True)

# convert data type to datetime format 
data['DateTime'] =  pd.to_datetime(data['DateTime'], infer_datetime_format=True)
data['Date'] = pd.to_datetime(data['Date']).dt.date
# recheck the data type 
print(data.dtypes)
print(data.head(10))

# data.rename(columns={'Systeemlijnnr': 'System_linenr'}, inplace=True)
# data = data[data['transtype']==30]
# # line 0 does not exist 
# data = data[data['System_linenr']!=0]

# Group data per datetime and line number 
grouped_data = pd.DataFrame(data.groupby(['DateTime', 'Systeemlijnnr'], as_index=False)['IdFactRit'].count().rename(columns={'IdFactRit': 'Total_trips'}))

grouped_data['DateTime'] =  pd.to_datetime(grouped_data['DateTime'], infer_datetime_format=True)
# grouped_data['Date'] = pd.to_datetime(grouped_data['DateTime']).dt.date
grouped_data = grouped_data.sort_values('DateTime')

print(grouped_data.dtypes)
print(grouped_data.head(10))

grouped_data['Total_trips'] = grouped_data['Total_trips'].astype(float)
# grouped_data['System_linenr'] = grouped_data['Publiekslijnnummer'].astype(float)
# grouped_data.drop('Systen_linenr', inplace=True)
# change the location of columns 
# cols = grouped_data.columns.tolist()
# print(cols)
# cols = ['DateTime', 'System_linenr', 'Total_trips']
# grouped_data = grouped_data[cols]

#%%
# Export cleaned data 

grouped_data.to_excel(r'C:/Users/FarahmandZH/OneDrive - Universiteit Twente/Documenten/PDEng project/Data/2021 test/trip_data2021.xlsx')

"""
Well done
"""





pleas


















