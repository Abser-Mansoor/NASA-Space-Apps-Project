# Imports
import os
import re
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from calendar import monthrange
from opendap_download.multi_processing_download import DownloadManager
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

####### INPUTS - CHANGE THESE #########
username = 'abser'
password = 'Absermansoor2@'
years = [2020, 2021, 2022, 2023, 2024]

# Define all variables needed for ML model
variables = [
    {
        'field_id': 'V10M',
        'field_name': 'V10M',
        'database_id': 'tavg1_2d_slv_Nx',
        'database_name': 'M2T1NXSLV',
        'conversion_function': lambda x: x,  # No conversion needed
        'aggregator': 'mean'
    }
    # Add other variables as needed...
]

# Define cities with point coordinates (name, country, latitude, longitude)
regions = [
    # ("karachi", "pakistan", 24.8607, 67.0011),
    # ("lahore", "pakistan", 31.5204, 74.3587),
    # ("islamabad", "pakistan", 33.6844, 73.0479),
    # ("rawalpindi", "pakistan", 33.5651, 73.0169),
    ("peshawar", "pakistan", 34.0151, 71.5249),
    # ("quetta", "pakistan", 30.1798, 66.9750),
    # ("multan", "pakistan", 30.1575, 71.5249),
    ("hyderabad", "pakistan", 25.3960, 68.3578),
    # ("mumbai", "india", 19.0760, 72.8777),
    # ("delhi", "india", 28.7041, 77.1025),
    # ("kolkata", "india", 22.5726, 88.3639),
    # ("chennai", "india", 13.0827, 80.2707),
    # ("bengaluru", "india", 12.9716, 77.5946),
    # ("hyderabad", "india", 17.3850, 78.4867),
    # ("ahmedabad", "india", 23.0225, 72.5714),
    # ("pune", "india", 18.5204, 73.8567),
    # ("surat", "india", 21.1702, 72.8311),
    # ("jaipur", "india", 26.9124, 75.7873),
    # ("lucknow", "india", 26.8467, 80.9462),
    # ("patna", "india", 25.5941, 85.1376),
    # ("dhaka", "bangladesh", 23.8103, 90.4125),
    # ("chittagong", "bangladesh", 22.3569, 91.7832),
    # ("colombo", "sri lanka", 6.9271, 79.8612),
    # ("kandy", "sri lanka", 7.2906, 80.6337),
    # ("kathmandu", "nepal", 27.7172, 85.3240),
]

####### CONSTANTS - DO NOT CHANGE BELOW THIS LINE #######
lat_coords = np.arange(0, 361, dtype=int)
lon_coords = np.arange(0, 576, dtype=int)
NUMBER_OF_CONNECTIONS = 5

####### HELPER FUNCTIONS - MUST INCLUDE THESE #######
def translate_lat_to_geos5_native(latitude):
    """
    The source for this formula is in the MERRA2 
    Variable Details - File specifications for GEOS pdf file.
    The Grid in the documentation has points from 1 to 361 and 1 to 576.
    The MERRA-2 Portal uses 0 to 360 and 0 to 575.
    latitude: float Needs +/- instead of N/S
    """
    return ((latitude + 90) / 0.5)

def translate_lon_to_geos5_native(longitude):
    """See function above"""
    return ((longitude + 180) / 0.625)

def find_closest_coordinate(calc_coord, coord_array):
    """
    Since the resolution of the grid is 0.5 x 0.625, the 'real world'
    coordinates will not be matched 100% correctly. This function matches 
    the coordinates as close as possible. 
    """
    # np.argmin() finds the smallest value in an array and returns its
    # index. np.abs() returns the absolute value of each item of an array.
    # To summarize, the function finds the difference closest to 0 and returns 
    # its index. 
    index = np.abs(coord_array-calc_coord).argmin()
    return coord_array[index]

def translate_year_to_file_number(year):
    """
    The file names consist of a number and a meta data string. 
    The number changes over the years. 1980 until 1991 it is 100, 
    1992 until 2000 it is 200, 2001 until 2010 it is  300 
    and from 2011 until now it is 400.
    """
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

def generate_url_params(parameter, time_para, lat_para, lon_para):
    """Creates a string containing all the parameters in query form"""
    parameter = map(lambda x: x + time_para, parameter)
    parameter = map(lambda x: x + lat_para, parameter)
    parameter = map(lambda x: x + lon_para, parameter)
    return ','.join(parameter)
    
def generate_download_links(download_years, base_url, dataset_name, url_params):
    """
    Generates the links for the download. 
    download_years: The years you want to download as array. 
    dataset_name: The name of the data set. For example tavg1_2d_slv_Nx
    """
    urls = []
    for y in download_years: 
        y_str = str(y)
        file_num = translate_year_to_file_number(y)
        for m in range(1,13):
            m_str = str(m).zfill(2)
            _, nr_of_days = monthrange(y, m)
            for d in range(1,nr_of_days+1):
                d_str = str(d).zfill(2)
                # Create the file name string
                file_name = 'MERRA2_{num}.{name}.{y}{m}{d}.nc4'.format(
                    num=file_num, name=dataset_name, 
                    y=y_str, m=m_str, d=d_str)
                # Create the query
                query = '{base}{y}/{m}/{name}.nc4?{params}'.format(
                    base=base_url, y=y_str, m=m_str, 
                    name=file_name, params=url_params)
                urls.append(query)
    return urls

def extract_date(data_set):
    """
    Extracts the date from the filename before merging the datasets. 
    """ 
    if 'HDF5_GLOBAL.Filename' in data_set.attrs:
        f_name = data_set.attrs['HDF5_GLOBAL.Filename']
    elif 'Filename' in data_set.attrs:
        f_name = data_set.attrs['Filename']
    else: 
        raise AttributeError('The attribute name has changed again!')
    # find a match between "." and ".nc4" that does not have "." .
    exp = r'(?<=\.)[^\.]*(?=\.nc4)'
    res = re.search(exp, f_name).group(0)
    # Extract the date. 
    y, m, d = res[0:4], res[4:6], res[6:8]
    date_str = ('%s-%s-%s' % (y, m, d))
    data_set = data_set.assign(date=date_str)
    return data_set

####### DOWNLOAD DATA #########
print('DOWNLOADING DATA FROM MERRA-2 FOR SUBCONTINENTAL REGION')
print('=====================')

# Download each variable separately for each city point
for var in variables:
    field_id = var['field_id']
    field_name = var['field_name']
    database_name = var['database_name']
    database_id = var['database_id']
    
    database_url = f'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/{database_name}.5.12.4/'
    
    print(f'Downloading {field_name} ({field_id}) data for subcontinental cities')
    
    # POINT-BASED EXTRACTION (like the reference script)
    for region_name, country, latitude, longitude in regions:
        print(f'  Processing {region_name}...')
        print(f'  Location: {latitude}°N, {longitude}°E')
        
        # Create directory for this variable and region
        os.makedirs(f'{field_name}/{region_name}', exist_ok=True)
        
        # POINT-BASED LOGIC: Translate single coordinate point
        lat_coord = translate_lat_to_geos5_native(latitude)
        lon_coord = translate_lon_to_geos5_native(longitude)
        
        # Find the closest coordinate in the grid
        lat_closest = find_closest_coordinate(lat_coord, lat_coords)
        lon_closest = find_closest_coordinate(lon_coord, lon_coords)
        
        print(f'  Grid coordinate - Lat: {lat_closest}, Lon: {lon_closest}')
        
        # Generate URLs for single point extraction
        requested_lat = f'[{lat_closest}:1:{lat_closest}]'  # Single point
        requested_lon = f'[{lon_closest}:1:{lon_closest}]'  # Single point
        
        parameter = generate_url_params([field_id], '[0:1:23]', requested_lat, requested_lon)
        generated_URL = generate_download_links(years, database_url, database_id, parameter)
        
        # Download data
        download_manager = DownloadManager()
        download_manager.set_username_and_password(username, password)
        download_manager.download_path = f'{field_name}/{region_name}'
        download_manager.download_urls = generated_URL
        download_manager.start_download(NUMBER_OF_CONNECTIONS)

print('FINISHED DOWNLOADING')
print('Files created in respective variable/city directories')