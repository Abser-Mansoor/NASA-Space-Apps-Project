import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_climatology(monthly_df, window=15):
    """
    Calculate daily climatology from monthly data using rolling window
    """
    # Ensure we're working with numeric data only (exclude date column)
    numeric_columns = [col for col in monthly_df.columns if col != 'date']
    monthly_numeric = monthly_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    
    # Set date as index for the numeric data
    monthly_indexed = monthly_numeric.copy()
    monthly_indexed.index = pd.to_datetime(monthly_df['date'])
    
    # Create a basic daily series by linear interpolation
    daily_index = pd.date_range(
        start=monthly_indexed.index.min(), 
        end=monthly_indexed.index.max(), 
        freq='D'
    )
    df_daily = monthly_indexed.reindex(daily_index).interpolate(method='linear')
    
    climatology = {}
    
    for column in df_daily.columns:
        doy_stats = []
        
        for doy in range(1, 367):  # Day of year 1-366
            window_values = []
            
            # Collect values from +/- window days around this DOY across all years
            for year in df_daily.index.year.unique():
                target_date = pd.Timestamp(f'{year}-01-01') + timedelta(days=doy-1)
                start_date = target_date - timedelta(days=window)
                end_date = target_date + timedelta(days=window)
                
                # Get values in the window
                window_data = df_daily.loc[
                    (df_daily.index >= start_date) & 
                    (df_daily.index <= end_date), 
                    column
                ].dropna()
                
                if len(window_data) > 0:
                    window_values.extend(window_data.values)
            
            if window_values:
                doy_stats.append({
                    'doy': doy,
                    'mean': float(np.mean(window_values)),
                    'std': float(np.std(window_values)) if len(window_values) > 1 else 0.1,
                    'min': float(np.min(window_values)),
                    'max': float(np.max(window_values))
                })
            else:
                # Fallback values if no data
                doy_stats.append({
                    'doy': doy,
                    'mean': 0.0,
                    'std': 0.1,
                    'min': -1.0,
                    'max': 1.0
                })
        
        climatology[column] = pd.DataFrame(doy_stats).set_index('doy')
    
    return climatology

def create_daily_from_climatology(monthly_df, climatology, variability_scale=0.3):
    """
    Create realistic daily values using climatology patterns
    """
    # Extract numeric columns and set date as index
    numeric_columns = [col for col in monthly_df.columns if col != 'date']
    monthly_numeric = monthly_df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    monthly_indexed = monthly_numeric.copy()
    monthly_indexed.index = pd.to_datetime(monthly_df['date'])
    
    # Create complete daily date range
    start_date = monthly_indexed.index.min()
    end_date = monthly_indexed.index.max()
    daily_index = pd.date_range(start=start_date, end=end_date, freq='D')
    
    df_daily = pd.DataFrame(index=daily_index, columns=numeric_columns)
    
    # First pass: linear interpolation for baseline
    df_baseline = monthly_indexed.reindex(daily_index).interpolate(method='linear')
    
    for date in daily_index:
        doy = date.timetuple().tm_yday
        month = date.month
        year = date.year
        
        for column in numeric_columns:
            if doy in climatology[column].index:
                # Get climatological statistics for this day of year
                clim_mean = climatology[column].loc[doy, 'mean']
                clim_std = climatology[column].loc[doy, 'std']
                
                # Get the monthly average for this specific month and year
                monthly_mask = (monthly_indexed.index.month == month) & (monthly_indexed.index.year == year)
                monthly_values = monthly_indexed.loc[monthly_mask, column]
                
                if len(monthly_values) > 0:
                    monthly_value = monthly_values.iloc[0]
                else:
                    # If no exact monthly match, use baseline
                    monthly_value = df_baseline.loc[date, column]
                
                # Calculate the typical value for this DOY
                typical_value = clim_mean
                
                # Calculate anomaly from climatology
                clim_anomaly = monthly_value - typical_value
                
                # Apply the anomaly to the climatological pattern
                daily_value = clim_mean + clim_anomaly
                
                # Add realistic daily variability (proportional to climatological std)
                if clim_std > 0:
                    daily_noise = np.random.normal(0, clim_std * variability_scale)
                    daily_value += daily_noise
                
                # Ensure we don't go beyond reasonable bounds
                min_bound = climatology[column].loc[doy, 'min']
                max_bound = climatology[column].loc[doy, 'max']
                daily_value = np.clip(daily_value, min_bound * 1.2, max_bound * 1.2)
                
                df_daily.loc[date, column] = daily_value
    
    # Fill any remaining gaps with linear interpolation
    df_daily = df_daily.astype(float).interpolate(method='linear')
    
    # Reset index to get date column back
    df_daily = df_daily.reset_index().rename(columns={'index': 'date'})
    
    return df_daily

def validate_daily_data(daily_df, monthly_df):
    """
    Validate that daily data preserves monthly statistics
    """
    print("\nVALIDATION RESULTS:")
    print("=" * 50)
    
    # Prepare data for comparison
    daily_indexed = daily_df.set_index('date')
    monthly_indexed = monthly_df.set_index('date')
    
    numeric_columns = [col for col in monthly_df.columns if col != 'date']
    
    for column in numeric_columns:
        # Resample daily to monthly
        daily_resampled = daily_indexed[column].resample('MS').mean()
        
        # Align dates for comparison
        common_dates = monthly_indexed.index.intersection(daily_resampled.index)
        
        if len(common_dates) > 0:
            monthly_original = monthly_indexed.loc[common_dates, column]
            monthly_from_daily = daily_resampled.loc[common_dates]
            
            # Calculate statistics
            corr = np.corrcoef(monthly_original, monthly_from_daily)[0, 1]
            rmse = np.sqrt(np.mean((monthly_original - monthly_from_daily) ** 2))
            bias = np.mean(monthly_from_daily - monthly_original)
            
            print(f"\n{column}:")
            print(f"  Correlation: {corr:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  Bias: {bias:.4f}")
            print(f"  Monthly mean - Original: {monthly_original.mean():.4f}, Daily-derived: {monthly_from_daily.mean():.4f}")

def process_noaa_daily_climatology(monthly_csv_path, variability_scale=0.3):
    """
    Main function to convert monthly NOAA indices to realistic daily values using climatology
    """
    # Load the monthly data
    monthly_df = pd.read_csv(monthly_csv_path, parse_dates=['date'])
    
    print("CLIMATOLOGY-BASED DAILY DOWNSCALING")
    print("=" * 50)
    print(f"Original monthly data:")
    print(f"  Date range: {monthly_df['date'].min()} to {monthly_df['date'].max()}")
    print(f"  Records: {len(monthly_df)}")
    numeric_columns = [col for col in monthly_df.columns if col != 'date']
    print(f"  Variables: {numeric_columns}")
    
    # Ensure numeric columns are properly typed
    for col in numeric_columns:
        monthly_df[col] = pd.to_numeric(monthly_df[col], errors='coerce')
    
    # Calculate climatology
    print("\nCalculating daily climatology...")
    climatology = calculate_climatology(monthly_df)
    
    print("Climatology statistics:")
    for column in climatology:
        print(f"  {column}: {len(climatology[column])} DOY records, mean std: {climatology[column]['std'].mean():.4f}")
    
    # Create daily values using climatology
    print(f"\nGenerating daily values (variability scale: {variability_scale})...")
    daily_df = create_daily_from_climatology(monthly_df, climatology, variability_scale)
    
    print(f"\nGenerated daily data:")
    print(f"  Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    print(f"  Records: {len(daily_df)}")
    print(f"  Missing values: {daily_df[numeric_columns].isnull().sum().sum()}")
    
    # Validate the results
    validate_daily_data(daily_df, monthly_df)
    
    return {
        'daily': daily_df,
        'monthly': monthly_df,
        'climatology': climatology
    }

if __name__ == "__main__":
    # Process with climatology method
    results = process_noaa_daily_climatology("NOAA_indices.csv", variability_scale=0.3)
    
    # Save the daily data
    daily_df = results['daily']
    daily_df.to_csv("NOAA_indices_daily_climatology.csv", index=False)
    
    print(f"\n" + "="*50)
    print(f"SAVED: NOAA_indices_daily_climatology.csv")
    print(f"Records: {len(daily_df)}")
    print(f"Date range: {daily_df['date'].min()} to {daily_df['date'].max()}")
    
    # Show sample of the data
    print(f"\nFirst 10 daily records:")
    print(daily_df.head(10))
    
    # Show summary statistics
    numeric_columns = [col for col in daily_df.columns if col != 'date']
    print(f"\nDaily data statistics:")
    print(daily_df[numeric_columns].describe())