import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import ftplib
from io import BytesIO, StringIO
import json
import re

def download_sst_daily():
    """Download daily sea surface temperature data"""
    print("Downloading Sea Surface Temperature (SST) data...")
    
    url = "https://climatereanalyzer.org/clim/sst_daily/json_2clim/oisst2.1_nino3.4_sst_day.json"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        sst_data = []
        current_date = datetime(2020, 1, 1)
        
        for year_data in data:
            year_str = year_data['name']
            # Stop when we encounter "Preliminary" or non-year entries
            if year_str == "2025":
                break
            try:
                year = int(year_str)
                if 2020 <= year <= 2024:  
                    daily_values = year_data['data']
                    
                    for day_idx, value in enumerate(daily_values):
                        if value is not None:
                            date = datetime(year, 1, 1) + timedelta(days=day_idx)
                            sst_data.append({
                                'date': date,
                                'SST': value
                            })
            except ValueError:
                # Skip entries that aren't valid years
                continue
        
        df = pd.DataFrame(sst_data)
        df = df.sort_values('date').reset_index(drop=True)
        print(f"  Downloaded {len(df)} daily SST records")
        return df
        
    except Exception as e:
        print(f"Error downloading SST: {e}")
        return pd.DataFrame(columns=['date', 'SST'])

def download_iod_weekly():
    """Download weekly Indian Ocean Dipole data via BOM's FTP service"""
    print("Downloading Indian Ocean Dipole (IOD) weekly data via FTP...")
    
    # BOM's anonymous FTP service
    ftp_host = "ftp.bom.gov.au"
    
    try:
        # First, let's explore the FTP directory structure to find the correct path
        print("  Exploring FTP directory structure...")
        ftp = ftplib.FTP(ftp_host)
        ftp.login()  # Anonymous login
        
        # List directories to find the correct path
        directories = [
            "anon/home/ncc/www/sco/iod/",
            "anon/gen/clim_data/",
            "anon/gen/ocprev/",
            "anon/gen/ocean/",
            "anon/home/ncc/"
        ]
        
        for directory in directories:
            try:
                print(f"  Checking {directory}...")
                file_list = []
                ftp.retrlines(f"LIST {directory}", file_list.append)
                print(f"    Found {len(file_list)} items")
                
                # Look for IOD files
                for item in file_list:
                    if 'iod' in item.lower():
                        print(f"    Found IOD file: {item}")
            except Exception as e:
                print(f"    Cannot access {directory}: {e}")
        
        # Try known IOD file paths
        iod_paths = [
            "anon/home/ncc/www/sco/iod/iod_1.txt",
            "anon/gen/clim_data/IDCK000072/iod_1.txt",
            "anon/home/ncc/www/sco/iod_1.txt",
            "anon/gen/ocprev/iod_1.txt"
        ]
        
        for iod_path in iod_paths:
            try:
                print(f"  Trying to download: {iod_path}")
                file_data = BytesIO()
                ftp.retrbinary(f"RETR {iod_path}", file_data.write)
                
                file_data.seek(0)
                text_content = file_data.read().decode('utf-8')
                
                if len(text_content.strip()) > 0:
                    print(f"  Successfully downloaded from: {iod_path}")
                    ftp.quit()
                    return process_iod_data(text_content)
                    
            except Exception as e:
                print(f"    Failed: {e}")
                continue
        
        ftp.quit()
        
    except Exception as e:
        print(f"Error with FTP: {e}")
    
    # If FTP fails, use the direct HTTP approach with the data you provided
    print("  FTP methods failed, using direct data processing...")
    return process_direct_iod_data()

def process_iod_data(text_content):
    """Process IOD data from text content"""
    iod_weekly_data = []
    lines = text_content.split('\n')
    
    for line in lines:
        line = line.strip()
        if line and ',' in line:
            parts = line.split(',')
            if len(parts) >= 3:
                try:
                    start_date_str = parts[0].strip()
                    end_date_str = parts[1].strip()
                    
                    if not (start_date_str.isdigit() and len(start_date_str) == 8 and
                            end_date_str.isdigit() and len(end_date_str) == 8):
                        continue

                    start_year = int(start_date_str[0:4])
                    start_month = int(start_date_str[4:6])
                    start_day = int(start_date_str[6:8])
                    
                    end_year = int(end_date_str[0:4])
                    end_month = int(end_date_str[4:6])
                    end_day = int(end_date_str[6:8])
                    
                    start_date = datetime(start_year, start_month, start_day)
                    end_date = datetime(end_year, end_month, end_day)
                    iod_value = float(parts[2])
                    
                    if 2020 <= start_year <= 2024:
                        mid_date = start_date + (end_date - start_date) / 2
                        iod_weekly_data.append({
                            'date': mid_date,
                            'IOD': iod_value
                        })
                except (ValueError, IndexError) as e:
                    continue
    
    weekly_df = pd.DataFrame(iod_weekly_data)
    if not weekly_df.empty:
        weekly_df = weekly_df.sort_values('date').reset_index(drop=True)
    print(f"  Processed {len(weekly_df)} IOD records")
    return weekly_df

def process_direct_iod_data():
    """Process the IOD data directly using the dataset you provided"""
    print("Processing IOD data from provided dataset...")
    
    # Your provided IOD dataset
    iod_data = """20200106,20200112,0.67
20200113,20200119,0.47
20200120,20200126,0.09
20200127,20200202,0.12
20200203,20200209,0.36
20200210,20200216,0.38
20200217,20200223,0.17
20200224,20200301,0.07
20200302,20200308,0.2
20200309,20200315,-0.07
20200316,20200322,-0.06
20200323,20200329,0.2
20200330,20200405,0.02
20200406,20200412,0.08
20200413,20200419,0.08
20200420,20200426,0.0
20200427,20200503,0.13
20200504,20200510,0.27
20200511,20200517,0.59
20200518,20200524,0.5
20200525,20200531,0.61
20200601,20200607,0.87
20200608,20200614,1.03
20200615,20200621,0.95
20200622,20200628,0.75
20200629,20200705,0.68
20200706,20200712,0.55
20200713,20200719,0.44
20200720,20200726,0.15
20200727,20200802,0.29
20200803,20200809,-0.02
20200810,20200816,-0.73
20200817,20200823,-0.96
20200824,20200830,-0.79
20200831,20200906,-0.43
20200907,20200913,-0.47
20200914,20200920,-0.59
20200921,20200927,-0.82
20200928,20201004,-0.62
20201005,20201011,-0.02
20201012,20201018,0.08
20201019,20201025,-0.27
20201026,20201101,-0.33
20201102,20201108,-0.18
20201109,20201115,-0.14
20201116,20201122,-0.34
20201123,20201129,-0.24
20201130,20201206,-0.01
20201207,20201213,0.03
20201214,20201220,0.06
20201221,20201227,0.15
20201228,20210103,0.31
20210104,20210110,0.06
20210111,20210117,-0.21
20210118,20210124,-0.1
20210125,20210131,0.1
20210201,20210207,0.13
20210208,20210214,-0.04
20210215,20210221,-0.03
20210222,20210228,0.02
20210301,20210307,0.0
20210308,20210314,0.14
20210315,20210321,0.15
20210322,20210328,0.06
20210329,20210404,0.04
20210405,20210411,0.21
20210412,20210418,0.21
20210419,20210425,0.27
20210426,20210502,0.42
20210503,20210509,0.48
20210510,20210516,0.15
20210517,20210523,-0.51
20210524,20210530,-0.72
20210531,20210606,-0.51
20210607,20210613,-0.36
20210614,20210620,-0.42
20210621,20210627,-0.41
20210628,20210704,-0.13
20210705,20210711,-0.24
20210712,20210718,-0.46
20210719,20210725,-0.78
20210726,20210801,-0.9
20210802,20210808,-0.82
20210809,20210815,-0.8
20210816,20210822,-0.7
20210823,20210829,-0.62
20210830,20210905,-0.69
20210906,20210912,-0.68
20210913,20210919,-0.83
20210920,20210926,-0.67
20210927,20211003,-0.91
20211004,20211010,-0.89
20211011,20211017,-0.75
20211018,20211024,-0.98
20211025,20211031,-0.68
20211101,20211107,-0.58
20211108,20211114,-0.45
20211115,20211121,-0.56
20211122,20211128,-0.45
20211129,20211205,-0.29
20211206,20211212,-0.26
20211213,20211219,-0.31
20211220,20211226,-0.06
20211227,20220102,0.07
20220103,20220109,0.06
20220110,20220116,0.03
20220117,20220123,-0.15
20220124,20220130,-0.56
20220131,20220206,-0.79
20220207,20220213,-0.46
20220214,20220220,-0.32
20220221,20220227,-0.37
20220228,20220306,-0.16
20220307,20220313,-0.12
20220314,20220320,-0.04
20220321,20220327,-0.05
20220328,20220403,-0.17
20220404,20220410,-0.08
20220411,20220417,-0.15
20220418,20220424,-0.21
20220425,20220501,0.02
20220502,20220508,0.12
20220509,20220515,0.16
20220516,20220522,-0.3
20220523,20220529,-0.26
20220530,20220605,-0.03
20220606,20220612,-0.03
20220613,20220619,-0.17
20220620,20220626,-0.23
20220627,20220703,-0.45
20220704,20220710,-1.02
20220711,20220717,-1.08
20220718,20220724,-1.09
20220725,20220731,-0.8
20220801,20220807,-0.88
20220808,20220814,-1.21
20220815,20220821,-1.37
20220822,20220828,-1.19
20220829,20220904,-1.09
20220905,20220911,-1.42
20220912,20220918,-1.4
20220919,20220925,-1.25
20220926,20221002,-1.17
20221003,20221009,-1.16
20221010,20221016,-1.45
20221017,20221023,-1.34
20221024,20221030,-1.03
20221031,20221106,-0.71
20221107,20221113,-0.46
20221114,20221120,-0.18
20221121,20221127,-0.01
20221128,20221204,-0.13
20221205,20221211,0.04
20221212,20221218,0.19
20221219,20221225,0.17
20221226,20230101,0.42
20230102,20230108,0.58
20230109,20230115,0.33
20230116,20230122,0.01
20230123,20230129,0.22
20230130,20230205,0.35
20230206,20230212,0.38
20230213,20230219,0.25
20230220,20230226,0.13
20230227,20230305,0.63
20230306,20230312,0.5
20230313,20230319,0.48
20230320,20230326,0.4
20230327,20230402,0.44
20230403,20230409,0.42
20230410,20230416,0.16
20230417,20230423,0.18
20230424,20230430,0.14
20230501,20230507,-0.21
20230508,20230514,-0.33
20230515,20230521,0.1
20230522,20230528,0.39
20230529,20230604,0.61
20230605,20230611,0.6
20230612,20230618,0.49
20230619,20230625,0.28
20230626,20230702,0.11
20230703,20230709,-0.13
20230710,20230716,0.02
20230717,20230723,0.22
20230724,20230730,0.21
20230731,20230806,0.36
20230807,20230813,0.34
20230814,20230820,0.7
20230821,20230827,1.1
20230828,20230903,1.11
20230904,20230910,1.03
20230911,20230917,1.16
20230918,20230924,1.38
20230925,20231001,1.53
20231002,20231008,1.75
20231009,20231015,1.91
20231016,20231022,1.91
20231023,20231029,1.51
20231030,20231105,1.55
20231106,20231112,1.52
20231113,20231119,1.6
20231120,20231126,1.66
20231127,20231203,1.63
20231204,20231210,1.51
20231211,20231217,1.29
20231218,20231224,1.06
20231225,20231231,0.88
20240101,20240107,0.89
20240108,20240114,0.62
20240115,20240121,0.93
20240122,20240128,0.74
20240129,20240204,0.5
20240205,20240211,0.42
20240212,20240218,0.06
20240219,20240225,-0.22
20240226,20240303,-0.23
20240304,20240310,0.25
20240311,20240317,0.97
20240318,20240324,0.38
20240325,20240331,0.24
20240401,20240407,0.33
20240408,20240414,0.27
20240415,20240421,0.31
20240422,20240428,0.15
20240429,20240505,0.15
20240506,20240512,0.34
20240513,20240519,0.47
20240520,20240526,0.52
20240527,20240602,0.36
20240603,20240609,0.27
20240610,20240616,0.25
20240617,20240623,0.24
20240624,20240630,0.05
20240701,20240707,-0.19
20240708,20240714,-0.29
20240715,20240721,-0.33
20240722,20240728,-0.3
20240729,20240804,-0.15
20240805,20240811,0.05
20240812,20240818,0.33
20240819,20240825,0.16
20240826,20240901,0.15
20240902,20240908,0.03
20240909,20240915,-0.07
20240916,20240922,-0.03
20240923,20240929,-0.39
20240930,20241006,-0.35
20241007,20241013,-0.58
20241014,20241020,-0.93
20241021,20241027,-0.94
20241028,20241103,-0.7
20241104,20241110,-0.69
20241111,20241117,-0.73
20241118,20241124,-0.54
20241125,20241201,-0.19
20241202,20241208,-0.22
20241209,20241215,-0.15
20241216,20241222,-0.26
20241223,20241229,-0.3
20241230,20250105,-0.16"""
    
    return process_iod_data(iod_data)

def download_nao_daily():
    """Download daily North Atlantic Oscillation data"""
    print("Downloading North Atlantic Oscillation (NAO) data...")
    
    url = "https://downloads.psl.noaa.gov/Public/map/teleconnections/nao.reanalysis.t10trunc.1948-present.txt"
    
    try:
        response = requests.get(url)
        lines = response.text.split('\n')
        
        nao_data = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        year = int(parts[0])
                        month = int(parts[1])
                        day = int(parts[2])
                        nao_value = float(parts[3])
                        
                        date = datetime(year, month, day)
                        if 2020 <= year <= 2024:
                            nao_data.append({
                                'date': date,
                                'NAO': nao_value
                            })
                    except (ValueError, IndexError):
                        continue
        
        df = pd.DataFrame(nao_data)
        df = df.sort_values('date').reset_index(drop=True)
        print(f"  Downloaded {len(df)} daily NAO records")
        return df
        
    except Exception as e:
        print(f"Error downloading NAO: {e}")
        return pd.DataFrame(columns=['date', 'NAO'])

def download_ao_monthly():
    """Download monthly Arctic Oscillation data"""
    print("Downloading Arctic Oscillation (AO) monthly data...")
    
    url = "https://psl.noaa.gov/data/correlation/ao.data"
    
    try:
        response = requests.get(url)
        lines = response.text.split('\n')
        
        ao_data = []
        
        for line in lines:
            line = line.strip()
            # Skip header line and empty lines
            if not line or not line[0].isdigit():
                continue
                
            parts = line.split()
            if len(parts) >= 13:  # Year + 12 months
                try:
                    year = int(parts[0])
                    if 2020 <= year <= 2024:
                        monthly_values = [float(x) for x in parts[1:13]]
                        
                        for month, value in enumerate(monthly_values, 1):
                            # Use 15th of each month as monthly value
                            date = datetime(year, month, 15)
                            ao_data.append({
                                'date': date,
                                'AO': value
                            })
                except (ValueError, IndexError):
                    continue
        
        df = pd.DataFrame(ao_data)
        df = df.sort_values('date').reset_index(drop=True)
        print(f"  Downloaded {len(df)} monthly AO records")
        return df
        
    except Exception as e:
        print(f"Error downloading AO: {e}")
        return pd.DataFrame(columns=['date', 'AO'])

def create_climatology_daily(low_freq_df, column_name, window_days=15, variability_scale=0.3):
    """
    Create daily values from low-frequency data using climatology
    Works for both monthly (AO) and weekly (IOD) data
    """
    if low_freq_df.empty:
        return low_freq_df
    
    # First create basic daily series by linear interpolation
    start_date = low_freq_df['date'].min()
    end_date = low_freq_df['date'].max()
    daily_index = pd.date_range(start=start_date, end=end_date, freq='D')
    
    df_daily_base = low_freq_df.set_index('date').reindex(daily_index).interpolate(method='linear')
    
    # Calculate climatology from the interpolated data
    doy_stats = []
    for doy in range(1, 367):  # Day of year
        window_values = []
        
        # Collect values from +/- window_days around this DOY across all years
        for year in df_daily_base.index.year.unique():
            target_date = pd.Timestamp(f'{year}-01-01') + timedelta(days=doy-1)
            start_window = target_date - timedelta(days=window_days)
            end_window = target_date + timedelta(days=window_days)
            
            window_data = df_daily_base.loc[
                (df_daily_base.index >= start_window) & 
                (df_daily_base.index <= end_window), 
                column_name
            ].dropna()
            
            if len(window_data) > 0:
                window_values.extend(window_data.values)
        
        if window_values:
            doy_stats.append({
                'doy': doy,
                'mean': np.mean(window_values),
                'std': np.std(window_values) if len(window_values) > 1 else 0.1,
                'min': np.min(window_values),
                'max': np.max(window_values)
            })
        else:
            # Fallback values
            doy_stats.append({
                'doy': doy,
                'mean': 0,
                'std': 0.1,
                'min': -2,
                'max': 2
            })
    
    climatology_df = pd.DataFrame(doy_stats).set_index('doy')
    
    # Create daily values using climatology
    daily_dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    daily_values = []
    
    for date in daily_dates:
        doy = date.timetuple().tm_yday
        
        # Get the low-frequency value for this time period
        if column_name == 'AO':  # Monthly data
            # Find the monthly value for this date's month and year
            monthly_mask = (low_freq_df['date'].dt.month == date.month) & (low_freq_df['date'].dt.year == date.year)
            period_values = low_freq_df.loc[monthly_mask, column_name]
        else:  # IOD - Weekly data
            # Find the closest weekly value (within 7 days)
            date_range_start = date - timedelta(days=7)
            date_range_end = date + timedelta(days=7)
            period_mask = (low_freq_df['date'] >= date_range_start) & (low_freq_df['date'] <= date_range_end)
            period_values = low_freq_df.loc[period_mask, column_name]
        
        if len(period_values) > 0:
            # Use the closest value (first one in the filtered set)
            period_value = period_values.iloc[0]
        else:
            # If no exact match, use base interpolation
            period_value = df_daily_base.loc[date, column_name] if date in df_daily_base.index else 0
        
        # Get climatological statistics for this DOY
        clim_mean = climatology_df.loc[doy, 'mean']
        clim_std = climatology_df.loc[doy, 'std']
        
        # Calculate anomaly from climatology and apply to daily pattern
        clim_anomaly = period_value - clim_mean
        daily_value = clim_mean + clim_anomaly
        
        # Add realistic daily variability (proportional to climatological std)
        if clim_std > 0:
            daily_noise = np.random.normal(0, clim_std * variability_scale)
            daily_value += daily_noise
        
        # Ensure values stay within reasonable bounds
        min_bound = climatology_df.loc[doy, 'min']
        max_bound = climatology_df.loc[doy, 'max']
        daily_value = np.clip(daily_value, min_bound * 1.2, max_bound * 1.2)
        
        daily_values.append(daily_value)
    
    daily_df = pd.DataFrame({
        'date': daily_dates,
        column_name: daily_values
    })
    
    return daily_df

def merge_all_datasets():
    """Download and merge all datasets into a unified daily CSV"""
    print("DOWNLOADING AND PROCESSING ALL CLIMATE INDICES")
    print("=" * 60)
    
    # Download all datasets
    sst_df = download_sst_daily()
    iod_weekly_df = download_iod_weekly()
    nao_df = download_nao_daily()
    ao_monthly_df = download_ao_monthly()
    
    # Create daily values using climatology for both AO and IOD
    print("Creating daily AO values using climatology...")
    ao_daily_df = create_climatology_daily(ao_monthly_df, 'AO', variability_scale=0.3)
    print(f"  Created {len(ao_daily_df)} daily AO records")
    
    print("Creating daily IOD values using climatology...")
    iod_daily_df = create_climatology_daily(iod_weekly_df, 'IOD', variability_scale=0.2)
    print(f"  Created {len(iod_daily_df)} daily IOD records")
    
    # Create master date range
    start_date = '2020-01-01'
    end_date = '2024-12-31'
    master_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    master_df = pd.DataFrame({'date': master_dates})
    
    print(f"\nMerging datasets for {len(master_dates)} days...")
    
    # Merge all datasets
    master_df = master_df.merge(sst_df, on='date', how='left')
    master_df = master_df.merge(iod_daily_df, on='date', how='left')
    master_df = master_df.merge(nao_df, on='date', how='left')
    master_df = master_df.merge(ao_daily_df, on='date', how='left')
    
    # Sort by date
    master_df = master_df.sort_values('date').reset_index(drop=True)
    
    # Fill any small gaps with linear interpolation
    master_df['SST'] = master_df['SST'].interpolate(method='linear')
    master_df['IOD'] = master_df['IOD'].interpolate(method='linear')
    master_df['NAO'] = master_df['NAO'].interpolate(method='linear')
    master_df['AO'] = master_df['AO'].interpolate(method='linear')
    
    return master_df

def validate_final_dataset(df):
    """Validate the final merged dataset"""
    print("\nVALIDATING FINAL DATASET")
    print("=" * 40)
    
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    print("\nMissing values:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        print(f"  {col}: {count} missing")
    
    print("\nData statistics:")
    stats = df[['SST', 'IOD', 'NAO', 'AO']].describe()
    print(stats)
    
    # Check date coverage
    expected_days = (datetime(2024, 12, 31) - datetime(2020, 1, 1)).days + 1
    actual_days = len(df)
    print(f"\nDate coverage: {actual_days}/{expected_days} days ({actual_days/expected_days*100:.1f}%)")

if __name__ == "__main__":
    # Merge all datasets
    final_df = merge_all_datasets()
    
    # Validate the dataset
    validate_final_dataset(final_df)
    
    # Save to CSV
    output_file = "climate_indices_daily_2020_2024.csv"
    final_df.to_csv(output_file, index=False)
    
    print(f"\n✅ SAVED: {output_file}")
    print(f"Final dataset: {len(final_df)} daily records")
    
    # Show sample data
    print(f"\nSAMPLE DATA (first 10 records):")
    print(final_df.head(10))
    
    print(f"\nSAMPLE DATA (last 10 records):")
    print(final_df.tail(10))