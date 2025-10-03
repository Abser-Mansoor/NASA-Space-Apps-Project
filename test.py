import os
import re
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from calendar import monthrange
import glob

def count_by_region():
    """
    Count records grouped by region
    """
    fields = ['QV2M']
    
    region_counts = {}
    
    for field in fields:
        csv_files = glob.glob(f"{field}/*daily.csv")
        
        for file_path in csv_files:
            try:
                region_name = os.path.basename(file_path).replace('_daily.csv', '')
                df = pd.read_csv(file_path)
                
                if region_name not in region_counts:
                    region_counts[region_name] = 0
                
                region_counts[region_name] += len(df)
                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    print(f"\n{'='*50}")
    print("RECORDS BY REGION")
    print(f"{'='*50}")
    
    total = 0
    for region, count in sorted(region_counts.items()):
        print(f"{region:<20} : {count:>6} records")
        total += count
    
    print(f"{'Total':<20} : {total:>6} records")
    
    # Expected per region: ~1827 records (5 years × 365 days + leap days)
    expected_per_region = 1827
    print(f"\nExpected per region: ~{expected_per_region} records")
    
    return region_counts

if __name__ == "__main__":
    region_counts = count_by_region()
