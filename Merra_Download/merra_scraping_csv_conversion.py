import os
import re
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from calendar import monthrange

def extract_date_from_filename(filename):
    """
    Extract date from filename directly without opening the file.
    Format: MERRA2_400.tavg1_2d_slv_Nx.20200101.nc4
    Returns date string (YYYY-MM-DD) or None if extraction fails.
    """
    try:
        # Extract the date part from filename
        exp = r'\.(\d{8})\.nc4'
        res = re.search(exp, filename)
        if res:
            date_str = res.group(1)
            y, m, d = date_str[0:4], date_str[4:6], date_str[6:8]
            return f'{y}-{m}-{d}'
        return None
    except Exception as e:
        print(f"Error extracting date from filename {filename}: {e}")
        return None

def extract_date(data_set):
    """
    Extracts the date from the filename before merging the datasets. 
    Returns None if extraction fails.
    """ 
    try:
        if 'HDF5_GLOBAL.Filename' in data_set.attrs:
            f_name = data_set.attrs['HDF5_GLOBAL.Filename']
        elif 'Filename' in data_set.attrs:
            f_name = data_set.attrs['Filename']
        else: 
            print("Warning: Filename attribute not found in dataset")
            return None
            
        # find a match between "." and ".nc4" that does not have "." .
        exp = r'(?<=\.)[^\.]*(?=\.nc4)'
        res = re.search(exp, f_name)
        if not res:
            print(f"Warning: Could not extract date from filename: {f_name}")
            return None
            
        res = res.group(0)
        # Extract the date. 
        y, m, d = res[0:4], res[4:6], res[6:8]
        date_str = ('%s-%s-%s' % (y, m, d))
        data_set = data_set.assign(date=date_str)
        return data_set
    except Exception as e:
        print(f"Error extracting date from dataset: {e}")
        return None

def safe_open_dataset(file_path, field_name):
    """
    Safely opens a NetCDF file and extracts data with proper error handling.
    Returns DataFrame if successful, None if failed.
    """
    try:
        # Try to open the dataset
        ds = xr.open_dataset(file_path, engine="netcdf4")
        
        # Check if the required variable exists
        if field_name not in ds.variables:
            print(f"Warning: Variable {field_name} not found in {os.path.basename(file_path)}")
            ds.close()
            return None
        
        # Extract date safely
        ds_with_date = extract_date(ds)
        if ds_with_date is None:
            print(f"Warning: Could not extract date from {os.path.basename(file_path)}")
            ds.close()
            return None
        
        # Convert to dataframe
        df_region = ds_with_date.to_dataframe().reset_index()
        
        # Keep only useful columns - EXCLUDE source_file from the main data
        required_columns = ['time', field_name]
        if all(col in df_region.columns for col in required_columns):
            # Include spatial coordinates if available
            spatial_cols = []
            if 'lat' in df_region.columns:
                spatial_cols.append('lat')
            if 'lon' in df_region.columns:
                spatial_cols.append('lon')
            
            # Select only numeric columns for processing
            df_region = df_region[required_columns + spatial_cols]
            
            ds.close()
            return df_region
        else:
            print(f"Warning: Required columns not found in {os.path.basename(file_path)}")
            ds.close()
            return None
            
    except Exception as e:
        print(f"Error processing file {os.path.basename(file_path)}: {e}")
        return None

def create_null_data_for_corrupted_file(filename, field_name, hours=24):
    """
    Create null data for a corrupted file with proper timestamps.
    Returns DataFrame with null values for all hours of the day.
    """
    try:
        # Extract date from filename
        date_str = extract_date_from_filename(filename)
        if not date_str:
            print(f"Could not extract date from corrupted file: {filename}")
            return None
        
        # Create hourly timestamps for the entire day
        base_date = datetime.strptime(date_str, '%Y-%m-%d')
        timestamps = [base_date + timedelta(hours=h) for h in range(hours)]
        
        # Create DataFrame with null values
        null_data = pd.DataFrame({
            'time': timestamps,
            field_name: [np.nan] * hours
        })
        
        print(f"  Created null data for corrupted file: {filename} ({date_str})")
        return null_data
        
    except Exception as e:
        print(f"Error creating null data for {filename}: {e}")
        return None

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


regions = [
    ("karachi", "pakistan", 24.8607, 67.0011),
    # ("lahore", "pakistan", 31.5204, 74.3587),
    # ("islamabad", "pakistan", 33.6844, 73.0479),
    # ("rawalpindi", "pakistan", 33.5651, 73.0169),
    # ("peshawar", "pakistan", 34.0151, 71.5249),
    # ("quetta", "pakistan", 30.1798, 66.9750),
    # ("multan", "pakistan", 30.1575, 71.5249),
    # ("hyderabad", "pakistan", 25.3960, 68.3578),
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

variables = [
    {
        'field_id': 'T2M',
        'field_name': 'temperature',
        'database_name': 'M2T1NXSLV',
        'database_id': 'tavg1_2d_slv_Nx',
        'conversion_function': lambda x: x - 273.15,  # Convert Kelvin to Celsius
        'aggregator': 'mean'
    }
]

lat_coords = np.arange(0, 361, dtype=int)
lon_coords = np.arange(0, 576, dtype=int)
username = 'abser'
password = 'Absermansoor2@'
years = [2020, 2021, 2022, 2023, 2024]

######### PROCESS AND MERGE DATA ##########
print('PROCESSING AND MERGING REGIONAL DATA')
print('=====================')

# Process each variable and create regional datasets
all_regional_data = {}
successful_files = []
failed_files = []
corrupted_files_with_dates = []

for var in variables:
    field_id = var['field_id']
    field_name_display = var['field_name']
    conversion_function = var['conversion_function']
    aggregator = var['aggregator']
    
    print(f'Processing {field_name_display} data...')
    
    # Updated to unpack all region parameters
    for region_name, country, latitude, longitude in regions:
        dfs = []
        data_dir = f'{field_name_display}/{region_name}'
        
        if not os.path.exists(data_dir):
            print(f'No data found for {field_name_display} in {region_name}')
            continue
        
        print(f'  Scanning directory: {data_dir}')
        nc4_files = [f for f in os.listdir(data_dir) if f.endswith(".nc4")]
        print(f'  Found {len(nc4_files)} NetCDF files')
        
        for file in nc4_files:
            file_path = os.path.join(data_dir, file)
            df_region = safe_open_dataset(file_path, field_id)
            
            if df_region is not None:
                dfs.append(df_region)
                successful_files.append(file)
                print(f'     Successfully processed: {file}')
            else:
                # Create null data for corrupted file
                null_data = create_null_data_for_corrupted_file(file, field_id)
                if null_data is not None:
                    dfs.append(null_data)
                    corrupted_files_with_dates.append({
                        'filename': file,
                        'date': extract_date_from_filename(file),
                        'status': 'corrupted_replaced_with_null'
                    })
                    print(f'    Created null data for corrupted file: {file}')
                else:
                    failed_files.append(file)
                    print(f'    ✗ Failed to process and could not create null data: {file}')
        
        if dfs:
            print(f'  Combining {len(dfs)} datasets (successful + corrupted with nulls)...')
            df_hourly_all = pd.concat(dfs, ignore_index=True)
            
            # DEBUG: Check data types and columns
            print(f"  Columns in combined data: {df_hourly_all.columns.tolist()}")
            print(f"  Data types: {df_hourly_all.dtypes}")
            
            # Select only numeric columns for aggregation
            numeric_columns = df_hourly_all.select_dtypes(include=[np.number]).columns.tolist()
            print(f"  Numeric columns for aggregation: {numeric_columns}")
            
            # Calculate regional average (mean of all grid points)
            # Only aggregate numeric columns
            if numeric_columns:
                df_hourly_avg = df_hourly_all.groupby(['time'])[numeric_columns].mean().reset_index()
                
                # Apply conversion function to the target field
                if field_id in df_hourly_avg.columns:
                    df_hourly_avg[field_id] = df_hourly_avg[field_id].apply(conversion_function)
                
                # Ensure proper datetime format
                df_hourly_avg['datetime'] = pd.to_datetime(df_hourly_avg['time'])
                df_hourly_avg = df_hourly_avg.drop('time', axis=1)
                
                # ADD REGION METADATA to hourly data
                df_hourly_avg['region_name'] = region_name
                df_hourly_avg['country'] = country
                df_hourly_avg['latitude'] = latitude
                df_hourly_avg['longitude'] = longitude
                
                # Save regional average hourly data
                output_dir = f'{field_name_display}'
                os.makedirs(output_dir, exist_ok=True)
                output_file = f'{output_dir}/{region_name}_hourly_avg.csv'
                df_hourly_avg.to_csv(output_file, index=False)
                
                # Store for combined dataset
                if region_name not in all_regional_data:
                    all_regional_data[region_name] = {}
                all_regional_data[region_name][field_id] = df_hourly_avg.set_index('datetime')[field_id]
                
                print(f'  {region_name}: {len(df_hourly_avg)} hourly records processed')
                print(f'  Saved to: {output_file}')
                
                # Count null values
                null_count = df_hourly_avg[field_id].isna().sum()
                if null_count > 0:
                    print(f' Contains {null_count} null values from corrupted files')
            else:
                print(f'  No numeric columns found for aggregation in {region_name}')
        else:
            print(f'  No valid data could be processed for {region_name}')

# Print comprehensive summary of processing results
print('\n' + '='*50)
print('PROCESSING SUMMARY')
print('='*50)
print(f'Successful files: {len(successful_files)}')
print(f'Corrupted files replaced with null data: {len(corrupted_files_with_dates)}')
print(f'Completely failed files: {len(failed_files)}')
print(f'Total files processed: {len(successful_files) + len(corrupted_files_with_dates) + len(failed_files)}')

if corrupted_files_with_dates:
    print('\nCorrupted files replaced with null values:')
    for corrupted in corrupted_files_with_dates:
        print(f'  - {corrupted["filename"]} (Date: {corrupted["date"]})')

if failed_files:
    print('\nCompletely failed files (no date extracted):')
    for failed_file in failed_files:
        print(f'  - {failed_file}')

# Create combined dataset with all variables for the region
if all_regional_data:
    print('\nCREATING COMBINED REGIONAL DATASET')
    for region_name in all_regional_data.keys():
        # Find the region details from the regions list
        region_details = next((r for r in regions if r[0] == region_name), None)
        if region_details:
            region_name, country, latitude, longitude = region_details
        
        # Combine all variables into one dataframe
        combined_df = pd.DataFrame(all_regional_data[region_name])
        
        # ADD REGION METADATA to combined hourly data
        combined_df['region_name'] = region_name
        combined_df['country'] = country
        combined_df['latitude'] = latitude
        combined_df['longitude'] = longitude
        
        # Save combined datasets
        combined_df.to_csv(f'{region_name}_combined_hourly.csv')
        
        # Create daily aggregates
        daily_agg = {}
        for var in variables:
            field_id = var['field_id']
            aggregator = var['aggregator']
            if aggregator == 'sum':
                daily_agg[field_id] = 'sum'
            else:
                daily_agg[field_id] = 'mean'
        
        # Reset index to get datetime as column for daily aggregation
        combined_daily = combined_df.reset_index()
        combined_daily['date'] = pd.to_datetime(combined_daily['datetime']).dt.date
        combined_daily = combined_daily.groupby('date').agg(daily_agg).reset_index()
        
        # ADD REGION METADATA to daily data
        combined_daily['region_name'] = region_name
        combined_daily['country'] = country
        combined_daily['latitude'] = latitude
        combined_daily['longitude'] = longitude
        
        combined_daily.to_csv(f'{region_name}_combined_daily.csv', index=False)
        
        print(f"Created combined dataset for {region_name}")
        print(f"  Hourly records: {len(combined_df)}")
        print(f"  Daily records: {len(combined_daily)}")
        
        # Show null statistics
        for var in variables:
            field_id = var['field_id']
            if field_id in combined_df.columns:
                null_count = combined_df[field_id].isna().sum()
                if null_count > 0:
                    print(f" {field_id}: {null_count} null values ({null_count/len(combined_df)*100:.2f}%)")
else:
    print('\nNo valid data was processed. Cannot create combined datasets.')

print('\nFINISHED')
if successful_files or corrupted_files_with_dates:
    print('Files created:')
    print('- Individual variable files in their respective folders')
    print('- subcontinental_region_combined_hourly.csv')
    print('- subcontinental_region_combined_daily.csv')