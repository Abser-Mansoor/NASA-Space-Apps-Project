# Imports
import os
import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time
from calendar import monthrange
from opendap_download.multi_processing_download import DownloadManager
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

####### INPUTS - CHANGE THESE #########
username = 'abser' # Username for MERRA download account
password = 'Absermansoor2@' # Password for MERRA download account
years = [i+2014 for i in range(12)] # List of years for which data will be downloaded
field_id = ['PRECTOTCORR'] # ID of field in MERRA-2 - find ID here: https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf 
field_name = ['bias corrected total precipitation'] # Name of field to be stored with downloaded data (can use any name you like)
database_name = 'M2T1NXFLX' # Name of database in which field is stored, can be looked up by ID here: https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf 
database_id = 'tavg1_2d_flx_Nx' # ID of database database in which field is stored, also can be looked up by ID here: https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf 
locs = [('karachi', 24.8607, 67.0011)]
conversion_functions = [lambda x: x] # Unit conversion functions for each field
aggregators = ['mean'] # Method by which data will be aggregated over days and weeks. Can be "sum", "mean", "min", or "max"

####### CONSTANTS - DO NOT CHANGE BELOW THIS LINE #######
lat_coords = np.arange(0, 361, dtype=int)
lon_coords = np.arange(0, 576, dtype=int)
database_url = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/' + database_name + '.5.12.4/'
NUMBER_OF_CONNECTIONS = 5

####### DOWNLOAD DATA #########
# Translate lat/lon into coordinates that MERRA-2 understands
def translate_lat_to_geos5_native(latitude):
    return ((latitude + 90) / 0.5)

def translate_lon_to_geos5_native(longitude):
    return ((longitude + 180) / 0.625)

def find_closest_coordinate(calc_coord, coord_array):
    index = np.abs(coord_array-calc_coord).argmin()
    return coord_array[index]

def translate_year_to_file_number(year):
    file_number = ''
    
    if year >= 1980 and year < 1992:
        file_number = '100'
    elif year >= 1992 and year < 2001:
        file_number = '200'
    elif year >= 2001 and year < 2011:
        file_number = '300'
    elif year >= 2011:
        file_number = '400'
    else:
        raise Exception('The specified year is out of range.')
    return file_number

def generate_url_params(parameters, time_para, lat_para, lon_para):
    """Creates a string containing all the parameters in query form"""
    parameters = map(lambda x: x + time_para, parameters)
    parameters = map(lambda x: x + lat_para, parameters)
    parameters = map(lambda x: x + lon_para, parameters)
    return ','.join(parameters)
    
def generate_download_links(download_years, base_url, dataset_name, url_params):
    urls = []
    for y in download_years: 
        y_str = str(y)
        file_num = translate_year_to_file_number(y)
        for m in range(1,13):
            m_str = str(m).zfill(2)
            _, nr_of_days = monthrange(y, m)
            for d in range(1,nr_of_days+1):
                d_str = str(d).zfill(2)
                file_name = 'MERRA2_{num}.{name}.{y}{m}{d}.nc4'.format(
                    num=file_num, name=dataset_name, 
                    y=y_str, m=m_str, d=d_str)
                query = '{base}{y}/{m}/{name}.nc4?{params}'.format(
                    base=base_url, y=y_str, m=m_str, 
                    name=file_name, params=url_params)
                urls.append(query)
    return urls

print('DOWNLOADING DATA FROM MERRA')
print('Predicted time: ' + str(len(years)*len(locs)*len(field_id)*6) + ' minutes')
print('=====================')

# Download each field separately
for i, (field_id_val, field_name_val) in enumerate(zip(field_id, field_name)):
    print(f'Downloading {field_name_val} ({field_id_val}) data')
    
    for loc, lat, lon in locs:
        print(f'  Processing location: {loc}')
        
        # Translate coordinates
        lat_coord = translate_lat_to_geos5_native(lat)
        lon_coord = translate_lon_to_geos5_native(lon)
        lat_closest = find_closest_coordinate(lat_coord, lat_coords)
        lon_closest = find_closest_coordinate(lon_coord, lon_coords)
        
        # Generate URLs for scraping
        requested_lat = '[{lat}:1:{lat}]'.format(lat=lat_closest)
        requested_lon = '[{lon}:1:{lon}]'.format(lon=lon_closest)
        parameter = generate_url_params([field_id_val], '[0:1:23]', requested_lat, requested_lon)
        generated_URL = generate_download_links(years, database_url, database_id, parameter)
        
        # Create field-specific directory
        download_dir = f'{field_name_val}/{loc}'
        os.makedirs(download_dir, exist_ok=True)
        
        download_manager = DownloadManager()
        download_manager.set_username_and_password(username, password)
        download_manager.download_path = download_dir
        download_manager.download_urls = generated_URL
        download_manager.start_download(NUMBER_OF_CONNECTIONS)

######### OPEN, CLEAN, MERGE DATA AND WRITE CSVS ##########
def extract_date(data_set):
    if 'HDF5_GLOBAL.Filename' in data_set.attrs:
        f_name = data_set.attrs['HDF5_GLOBAL.Filename']
    elif 'Filename' in data_set.attrs:
        f_name = data_set.attrs['Filename']
    else: 
        raise AttributeError('The attribute name has changed again!')
    
    exp = r'(?<=\.)[^\.]*(?=\.nc4)'
    res = re.search(exp, f_name).group(0)
    y, m, d = res[0:4], res[4:6], res[6:8]
    date_str = ('%s-%s-%s' % (y, m, d))
    data_set = data_set.assign(date=date_str)
    return data_set

print('CLEANING AND MERGING DATA')
print('Predicted time: ' + str(len(years)*len(locs)*len(field_id)*0.1) + ' minutes')
print('=====================')

# Process each field separately
for i, (field_id_val, field_name_val, conversion_func, aggregator) in enumerate(zip(field_id, field_name, conversion_functions, aggregators)):
    print(f'Processing {field_name_val} ({field_id_val}) data')
    
    for loc, lat, lon in locs:
        print(f'  Cleaning and merging data for {loc}')
        
        download_dir = f'{field_name_val}/{loc}'
        dfs = []
        
        for file in os.listdir(download_dir):
            if '.nc4' in file:
                try:
                    with xr.open_mfdataset(download_dir + '/' + file, preprocess=extract_date) as df:
                        dfs.append(df.to_dataframe())
                except Exception as e:
                    print(f'Issue with file {file}: {e}')
        
        if not dfs:
            print(f'    WARNING: No data found for {field_name_val} in {loc}')
            continue
            
        df_hourly = pd.concat(dfs)
        df_hourly['time'] = df_hourly.index.get_level_values(level=2)
        
        # Handle column naming for multiple variables
        data_columns = [col for col in df_hourly.columns if col not in ['date', 'time']]
        if len(data_columns) == 1:
            df_hourly = df_hourly.rename(columns={data_columns[0]: field_name_val})
        else:
            print(f'    WARNING: Multiple data columns found: {data_columns}')
            # Use the first data column
            df_hourly = df_hourly.rename(columns={data_columns[0]: field_name_val})
            df_hourly = df_hourly[[field_name_val, 'date', 'time']]
        
        df_hourly[field_name_val] = df_hourly[field_name_val].apply(conversion_func)
        df_hourly['date'] = pd.to_datetime(df_hourly['date'])
        
        # Save hourly data
        hourly_file = f'{field_name_val}/{loc}_{field_id_val}_hourly.csv'
        df_hourly.to_csv(hourly_file, index=False)
        
        # Create daily aggregation
        df_daily = df_hourly.groupby('date').agg({field_name_val: aggregator})
        df_daily = df_daily.reset_index()
        daily_file = f'{field_name_val}/{loc}_{field_id_val}_daily.csv'
        df_daily.to_csv(daily_file, index=False)
        
        # Create weekly aggregation
        df_weekly = df_daily.copy()
        df_weekly['Year'] = pd.to_datetime(df_weekly['date']).dt.year
        df_weekly['Week'] = pd.to_datetime(df_weekly['date']).dt.isocalendar().week
        df_weekly = df_weekly.groupby(['Year', 'Week']).agg({field_name_val: aggregator})
        df_weekly = df_weekly.reset_index()
        weekly_file = f'{field_name_val}/{loc}_{field_id_val}_weekly.csv'
        df_weekly.to_csv(weekly_file, index=False)

print('FINISHED')