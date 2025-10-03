import os
import re
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from calendar import monthrange
import glob

# Alternative version without year column if you don't want it
def merge_all_fields_daily_no_year(output_filename="all_fields_daily_no_year.csv"):
    """
    Merge daily data without year column - just day_of_year and region.
    """
    
    all_fields = [
        # 'QV2M',
        # 'PS',
        # 'PRECTOTCORR',
        # 'T2M',
        # 'U10M',
        'V10M'
    ]
    output_filename = f"{all_fields[0]}.csv"
    all_rows = []
    
    for field in all_fields:
        daily_files = glob.glob(f"{field}/*_daily.csv")
        print(f"Processing {len(daily_files)} files for field: {field}")
        
        for file_path in daily_files:
            try:
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                
                for _, row in df.iterrows():
                    date = row['date']
                    lat = row['latitude']
                    lon = row['longitude']
                    
                    new_row = {
                        'date': date,
                        'latitude': lat,
                        'longitude': lon,
                        field: row[field]
                    }
                    
                    all_rows.append(new_row)
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    if not all_rows:
        print("No data found to merge.")
        return None
    
    final_df = pd.DataFrame(all_rows)
    final_df = final_df.sort_values(['date']).reset_index(drop=True)
    final_df.to_csv(output_filename, index=False)
    
    print(f"\nAll fields consolidated to: {output_filename}")
    print(f"Total records: {len(final_df)}")
    
    return final_df

# Usage examples:
if __name__ == "__main__":
   
    # Version without year column
    print("\nCreating consolidated file without year...")
    merged_no_year = merge_all_fields_daily_no_year(output_filename='temp_no_year.csv')