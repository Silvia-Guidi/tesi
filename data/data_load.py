import pandas as pd
import glob
import os

# 1. Setup paths (assuming files are in a folder named 'gen_data')
file_list = sorted(glob.glob('gen_data/*.csv'))

exo_list = []

# Types to include in the Exogenous shock matrix
target_types = [
    'Solar', 
    'Wind Onshore', 
    'Wind Offshore', 
    'Hydro Run-of-river and poundage'
]

print(f"Found {len(file_list)} files. Starting processing...")

for file in file_list:
    # ENTSO-E CSVs usually use TAB (\t) or Semicolon (;)
    df_temp = pd.read_csv(file, sep='\t')
    
    # Basic cleaning of column names (removes [MW] etc.)
    df_temp.columns = [c.split('[')[0].strip() for c in df_temp.columns]
    
    # Convert to datetime
    df_temp['DateTime(UTC)'] = pd.to_datetime(df_temp['DateTime(UTC)'])
    
    # Filter for the exogenous variables
    df_filtered = df_temp[df_temp['ProductionType'].isin(target_types)]
    
    # Pivot: Index=Time, Columns=[Country, Variable], Values=Generation
    # This creates the "Wide" structure per month
    month_pivot = df_filtered.pivot_table(
        index='DateTime(UTC)',
        columns=['AreaDisplayName', 'ProductionType'],
        values='ActualGenerationOutput',
        aggfunc='sum'
    )
    
    exo_list.append(month_pivot)
    print(f"Processed: {os.path.basename(file)}")

# 3. Concatenate all months into one full-year matrix
X_raw = pd.concat(exo_list).sort_index()

# 4. Resample to Daily (Summing MW to get Daily Energy Profile)
# This is crucial for BVARX to align with daily load/price data
X_daily = X_raw.resample('D').sum()

# 5. Final Cleaning for the Gibbs Sampler
# Fill NaNs with 0 (missing renewable data is treated as zero production)
X_daily = X_daily.fillna(0)

# Flatten columns to a single string (e.g., "Italy_Solar")
X_daily.columns = [f"{country}_{prod}" for country, prod in X_daily.columns]

print("\nExogenous Matrix (X) created successfully!")
print(f"Shape: {X_daily.shape} (Days x Variables)")