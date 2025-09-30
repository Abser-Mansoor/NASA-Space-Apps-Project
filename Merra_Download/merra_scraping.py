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
years = [2021, 2022, 2023, 2024, 2025]

# Define all variables needed for ML model
variables = [
    {
        'field_id': 'PRECTOTCORR',
        'field_name': 'precipitation',
        'database_name': 'M2T1NXFLX',
        'database_id': 'tavg1_2d_flx_Nx',
        'conversion_function': lambda x: x * 3600,  # Convert kg/m²/s to mm/hour
        'aggregator': 'sum'
    },
    {
        'field_id': 'T2M',
        'field_name': 'temperature',
        'database_name': 'M2T1NXSLV',
        'database_id': 'tavg1_2d_slv_Nx',
        'conversion_function': lambda x: x - 273.15,  # Convert Kelvin to Celsius
        'aggregator': 'mean'
    },
    {
        'field_id': 'QV2M',
        'field_name': 'humidity',
        'database_name': 'M2T1NXSLV',
        'database_id': 'tavg1_2d_slv_Nx',
        'conversion_function': lambda x: x * 1000,  # Convert kg/kg to g/kg
        'aggregator': 'mean'
    },
    {
        'field_id': 'PS',
        'field_name': 'surface_pressure',
        'database_name': 'M2T1NXSLV',
        'database_id': 'tavg1_2d_slv_Nx',
        'conversion_function': lambda x: x / 100,  # Convert Pa to hPa/mbar
        'aggregator': 'mean'
    },
    {
        'field_id': 'U10M',
        'field_name': 'u_wind',
        'database_name': 'M2T1NXSLV',
        'database_id': 'tavg1_2d_slv_Nx',
        'conversion_function': lambda x: x,  # No conversion needed
        'aggregator': 'mean'
    },
    {
        'field_id': 'V10M',
        'field_name': 'v_wind',
        'database_id': 'tavg1_2d_slv_Nx',
        'database_name': 'M2T1NXSLV',
        'conversion_function': lambda x: x,  # No conversion needed
        'aggregator': 'mean'
    }
]

# Define subcontinental region (covers Pakistan, India, Sri Lanka, Bangladesh)
# Coordinates: 21.9705° N to 5.9167° N, 60.8828° E to 74.5695° E
regions = [
    ('subcontinental_region', 21.9705, 5.9167, 60.8828, 74.5695),
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

# Download each variable separately for the region
for var in variables:
    field_id = var['field_id']
    field_name = var['field_name']
    database_name = var['database_name']
    database_id = var['database_id']
    
    database_url = f'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/{database_name}.5.12.4/'
    
    print(f'Downloading {field_name} ({field_id}) data for subcontinental region')
    
    for region_name, north_lat, south_lat, west_lon, east_lon in regions:
        print(f'  Processing {region_name}...')
        print(f'  Bounding box: {north_lat}°N to {south_lat}°N, {west_lon}°E to {east_lon}°E')
        
        # Create directory for this variable and region
        os.makedirs(f'{field_name}/{region_name}', exist_ok=True)
        
        # Translate the bounding box coordinates to grid coordinates
        north_lat_coord = translate_lat_to_geos5_native(north_lat)
        south_lat_coord = translate_lat_to_geos5_native(south_lat)
        west_lon_coord = translate_lon_to_geos5_native(west_lon)
        east_lon_coord = translate_lon_to_geos5_native(east_lon)
        
        # Find the closest coordinates in the grid
        north_lat_closest = find_closest_coordinate(north_lat_coord, lat_coords)
        south_lat_closest = find_closest_coordinate(south_lat_coord, lat_coords)
        west_lon_closest = find_closest_coordinate(west_lon_coord, lon_coords)
        east_lon_closest = find_closest_coordinate(east_lon_coord, lon_coords)
        
        # Ensure proper ordering (south to north, west to east)
        lat_start = min(south_lat_closest, north_lat_closest)
        lat_end = max(south_lat_closest, north_lat_closest)
        lon_start = min(west_lon_closest, east_lon_closest)
        lon_end = max(west_lon_closest, east_lon_closest)
        
        print(f'  Grid coordinates - Lat: {lat_start} to {lat_end}, Lon: {lon_start} to {lon_end}')
        
        # Generate URLs for the rectangular region
        requested_lat = f'[{lat_start}:1:{lat_end}]'
        requested_lon = f'[{lon_start}:1:{lon_end}]'
        
        parameter = generate_url_params([field_id], '[0:1:23]', requested_lat, requested_lon)
        generated_URL = generate_download_links(years, database_url, database_id, parameter)
        
        # Download data
        download_manager = DownloadManager()
        download_manager.set_username_and_password(username, password)
        download_manager.download_path = f'{field_name}/{region_name}'
        download_manager.download_urls = generated_URL
        download_manager.start_download(NUMBER_OF_CONNECTIONS)

######### PROCESS AND MERGE DATA ##########
print('PROCESSING AND MERGING REGIONAL DATA')
print('=====================')

# Process each variable and create regional datasets
all_regional_data = {}

for var in variables:
    field_name = var['field_name']
    conversion_function = var['conversion_function']
    aggregator = var['aggregator']
    
    print(f'Processing {field_name} data...')
    
    for region_name, north_lat, south_lat, west_lon, east_lon in regions:
        dfs = []
        data_dir = f'{field_name}/{region_name}'
        
        if not os.path.exists(data_dir):
            print(f'No data found for {field_name} in {region_name}')
            continue
            
        for file in os.listdir(data_dir):
            if '.nc4' in file:
                try:
                    with xr.open_mfdataset(data_dir + '/' + file, preprocess=extract_date) as df:
                        # Convert to dataframe with multiple grid points
                        df_region = df.to_dataframe()
                        dfs.append(df_region)
                except Exception as e:
                    print(f'Issue with file {file}: {e}')
        
        if dfs:
            df_hourly_all = pd.concat(dfs)
            
            # Calculate regional average (mean of all grid points)
            df_hourly_avg = df_hourly_all.groupby(['time', 'date']).mean().reset_index()
            df_hourly_avg.columns = [field_name, 'date', 'time']
            df_hourly_avg[field_name] = df_hourly_avg[field_name].apply(conversion_function)
            df_hourly_avg['date'] = pd.to_datetime(df_hourly_avg['date'])
            df_hourly_avg['datetime'] = df_hourly_avg['date'] + pd.to_timedelta(df_hourly_avg['time'].astype(str) + ':00:00')
            
            # Save regional average hourly data
            df_hourly_avg.to_csv(f'{field_name}/{region_name}_hourly_avg.csv', index=False)
            
            # Create daily aggregates
            df_daily = df_hourly_avg.groupby('date').agg(aggregator)
            df_daily = df_daily.drop('time', axis=1, errors='ignore')
            df_daily['date'] = df_daily.index
            df_daily.to_csv(f'{field_name}/{region_name}_daily_avg.csv', index=False)
            
            # Store for combined dataset
            if region_name not in all_regional_data:
                all_regional_data[region_name] = {}
            all_regional_data[region_name][field_name] = df_hourly_avg.set_index('datetime')[field_name]
            
            print(f'  {region_name}: {len(df_hourly_avg)} hourly records processed')

# Create combined dataset with all variables for the region
print('CREATING COMBINED REGIONAL DATASET')
for region_name in all_regional_data.keys():
    combined_df = pd.DataFrame(all_regional_data[region_name])
    
    # Add region metadata
    combined_df['region'] = region_name
    combined_df['north_lat'] = 21.9705
    combined_df['south_lat'] = 5.9167  
    combined_df['west_lon'] = 60.8828
    combined_df['east_lon'] = 74.5695
    
    # Save combined datasets
    combined_df.to_csv(f'subcontinental_region_combined_hourly.csv')
    
    # Create daily aggregates
    daily_agg = {}
    for var in variables:
        field_name = var['field_name']
        aggregator = var['aggregator']
        if aggregator == 'sum':
            daily_agg[field_name] = 'sum'
        else:
            daily_agg[field_name] = 'mean'
    
    combined_daily = combined_df.groupby(combined_df.index.date).agg(daily_agg)
    combined_daily['region'] = region_name
    combined_daily['north_lat'] = 21.9705
    combined_daily['south_lat'] = 5.9167
    combined_daily['west_lon'] = 60.8828
    combined_daily['east_lon'] = 74.5695
    combined_daily.to_csv('subcontinental_region_combined_daily.csv')
    
    print(f"Created combined dataset for {region_name}")
    print(f"  Hourly records: {len(combined_df)}")
    print(f"  Daily records: {len(combined_daily)}")

print('FINISHED')
print('Files created:')
print('- Individual variable files in their respective folders')
print('- subcontinental_region_combined_hourly.csv')
print('- subcontinental_region_combined_daily.csv')