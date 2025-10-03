import pandas as pd
import glob
import os

# Paths
features_path = r"C:\Laiba\Hackathon\NASA-Space-Apps-Project\Fields_Combined_Datasets"
climate_indices_path = r"C:\Laiba\Hackathon\NASA-Space-Apps-Project\Fields_Combined_Datasets\climate_indices_daily_2020_2024.csv"
output_path = r"C:\Laiba\Hackathon\NASA-Space-Apps-Project\master_inner.csv"

# Helper function to clean column names
def clean_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace('\ufeff', '')  # remove BOM if present
    return df

# Load all feature CSVs (excluding climate indices)
feature_files = [f for f in glob.glob(os.path.join(features_path, "*.csv")) 
                 if "climate_indices" not in f.lower()]
feature_dfs = []

for file in feature_files:
    df = pd.read_csv(file)
    df = clean_columns(df)

    # Rename variations to standard names
    df = df.rename(columns={
        'lat': 'latitude',
        'lon': 'longitude',
        'long': 'longitude'
    })

    # Ensure essential columns exist
    for col in ['date', 'latitude', 'longitude']:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in file {file}. Found columns: {df.columns.tolist()}")

    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])

    # Use filename as feature column name
    feature_name = os.path.splitext(os.path.basename(file))[0]
    if df.shape[1] > 3:  # assume 4th column is the feature value
        df = df.rename(columns={df.columns[3]: feature_name})

    feature_dfs.append(df)

# Merge all feature CSVs on ['date', 'latitude', 'longitude']
master_df = feature_dfs[0]
for df in feature_dfs[1:]:
    master_df = master_df.merge(df, on=['date', 'latitude', 'longitude'], how='inner')

# Load climate indices CSV
climate_df = pd.read_csv(climate_indices_path)
climate_df = clean_columns(climate_df)
climate_df['date'] = pd.to_datetime(climate_df['date'])

# Merge climate indices only on date
master_df = master_df.merge(climate_df, on="date", how="left")

# Save final master CSV
master_df.to_csv(output_path, index=False)

print("Master CSV created successfully:", output_path)
print(master_df.head())

# Ensure 'date' column is datetime
master_df['date'] = pd.to_datetime(master_df['date'])

# Add day of year column
master_df['DOY'] = master_df['date'].dt.dayofyear

master_df.to_csv(r"C:\Laiba\Hackathon\NASA-Space-Apps-Project\master.csv", index=False)

# Quick check
print(master_df[['date', 'DOY']].head(50))

