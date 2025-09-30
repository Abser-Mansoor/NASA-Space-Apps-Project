import pandas as pd
import numpy as np
from datetime import datetime
import requests

def filter_numerical_lines(text, expected_values):
    """Filter lines to only those with exactly expected_values numerical values"""
    filtered_lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        # Check if we have exactly expected_values parts and all are numerical
        if len(parts) == expected_values:
            numerical_count = 0
            for part in parts:
                try:
                    float(part)
                    numerical_count += 1
                except ValueError:
                    break
            
            # If all parts are numerical, keep this line
            if numerical_count == expected_values:
                filtered_lines.append(line)
    
    return filtered_lines

def load_ao():
    """Load Arctic Oscillation data - 13 values per line (year + 12 months)"""
    url = "https://psl.noaa.gov/data/correlation/ao.data"
    try:
        response = requests.get(url, timeout=10)
        # Filter for lines with exactly 13 numerical values
        filtered_lines = filter_numerical_lines(response.text, 13)
        
        if not filtered_lines:
            print("No valid AO data found")
            return pd.DataFrame(columns=["date", "AO"])
        
        # Parse the filtered data
        data = []
        for line in filtered_lines:
            parts = line.split()
            year = int(parts[0])
            for month in range(12):
                try:
                    value = float(parts[month + 1])
                    date = datetime(year, month + 1, 15)
                    data.append({'date': date, 'AO': value})
                except (ValueError, IndexError):
                    continue
        
        df = pd.DataFrame(data)
        return df.sort_values("date").reset_index(drop=True)
        
    except Exception as e:
        print(f"Error loading AO: {e}")
        return pd.DataFrame(columns=["date", "AO"])

def load_nao():
    """Load North Atlantic Oscillation data - 3 values per line (year, month, value)"""
    url = "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii"
    try:
        response = requests.get(url, timeout=10)
        # Filter for lines with exactly 3 numerical values
        filtered_lines = filter_numerical_lines(response.text, 3)
        
        if not filtered_lines:
            print("No valid NAO data found")
            return pd.DataFrame(columns=["date", "NAO"])
        
        # Parse the filtered data (year, month, value format)
        data = []
        for line in filtered_lines:
            parts = line.split()
            try:
                year = int(parts[0])
                month = int(parts[1])
                value = float(parts[2])
                date = datetime(year, month, 15)
                data.append({'date': date, 'NAO': value})
            except (ValueError, IndexError):
                continue
        
        df = pd.DataFrame(data)
        return df.sort_values("date").reset_index(drop=True)
        
    except Exception as e:
        print(f"Error loading NAO: {e}")
        return pd.DataFrame(columns=["date", "NAO"])

def load_iod():
    """Load Indian Ocean Dipole data - 13 values per line (year + 12 months)"""
    url = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"
    try:
        response = requests.get(url, timeout=10)
        # Filter for lines with exactly 13 numerical values
        filtered_lines = filter_numerical_lines(response.text, 13)
        
        if not filtered_lines:
            print("No valid IOD data found")
            return pd.DataFrame(columns=["date", "IOD"])
        
        # Parse the filtered data
        data = []
        for line in filtered_lines:
            parts = line.split()
            year = int(parts[0])
            for month in range(12):
                try:
                    value = float(parts[month + 1])
                    date = datetime(year, month + 1, 15)
                    data.append({'date': date, 'IOD': value})
                except (ValueError, IndexError):
                    continue
        
        df = pd.DataFrame(data)
        return df.sort_values("date").reset_index(drop=True)
        
    except Exception as e:
        print(f"Error loading IOD: {e}")
        return pd.DataFrame(columns=["date", "IOD"])

def get_noaa_indices():
    print("Downloading climate indices...")
    
    # Load each index
    ao = load_ao()
    nao = load_nao()
    iod = load_iod()
    
    print(f"AO records: {len(ao)} (expected: ~{12 * (datetime.now().year - 1950)})")
    print(f"NAO records: {len(nao)} (expected: ~{12 * (datetime.now().year - 1950)})")
    print(f"IOD records: {len(iod)} (expected: ~{12 * (datetime.now().year - 1870)})")
    
    # Merge all indices
    df = ao.merge(nao, on="date", how="outer")
    df = df.merge(iod, on="date", how="outer")
    df = df.sort_values("date").reset_index(drop=True)
    
    # Filter for last 10 years
    cutoff = datetime.now().year - 10
    df = df[df["date"].dt.year >= cutoff]
    
    # Remove any rows with all NaN values
    df = df.dropna(subset=["AO", "NAO", "IOD"], how="all")
    
    return df

def test_filtering():
    """Test the filtering on each data source"""
    urls = {
        "AO (13 values)": "https://psl.noaa.gov/data/correlation/ao.data",
        "NAO (3 values)": "https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii",
        "IOD (13 values)": "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data"
    }
    
    for name, url in urls.items():
        print(f"\n=== Testing {name} ===")
        try:
            response = requests.get(url, timeout=10)
            expected_values = 13 if "13" in name else 3
            filtered_lines = filter_numerical_lines(response.text, expected_values)
            
            print(f"Total lines: {len(response.text.splitlines())}")
            print(f"Filtered lines: {len(filtered_lines)}")
            if filtered_lines:
                print(f"First filtered line: {filtered_lines[0]}")
                print(f"Last filtered line: {filtered_lines[-1]}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Test the filtering first
    test_filtering()
    
    print("\n" + "="*60 + "\n")
    
    # Then download and process
    df = get_noaa_indices()
    
    print(f"\nFinal dataset: {len(df)} records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print("\nLast 10 records:")
    print(df.tail(10))
    
    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    
    # Save to CSV
    df.to_csv("NOAA_indices.csv", index=False)
    print(f"\nSaved to NOAA_indices.csv")