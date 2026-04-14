"""
INPUT FILES (xlsx):
    gen_data            -> Renewable generation (wind, solar, hydro) per country
    load_data           -> Actual total load per country
    price_data          -> Electricity prices (intraday or day-ahead) per country
    reserve_margin_data -> Reserve margin per country

COLUMN CONVENTION in each file:
    {COUNTRY}_{variable}  e.g., "IT_Wind", "DE_Solar", "FR_Load", "IT_Price"

OUTPUT:
    Y  -> DataFrame [T x n_endo]   Endogenous variables (Price, Load)
    X  -> DataFrame [T x n_exo]    Exogenous variables (Wind, Solar, Hydro, CO2, EPU, ReserveMargin)
    df -> DataFrame [T x (n_endo + n_exo)]  Complete dataset for inspection


"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline

#import params

DATA_DIR = Path(__file__).parent  

FILE_MAP = {
    "gen"     : DATA_DIR /"gen_data"/ "energy_pivot.xlsx",
    "load"    : DATA_DIR / "load_data"/"load_pivot.xlsx",
    "price"   : DATA_DIR / "price_data"/"prices_pivot.xlsx",
    "reserve" : DATA_DIR / "reserve_margin_data"/"reserve_margin_data.xlsx",
}

# Countries to Inlude in the Analysis
COUNTRIES = ["AT", "BA", "BE", "BG", "CH", "CZ", "DE",
    "DK", "EE", "ES", "FI", "FR",
    "GR", "HR", "HU", "IE", "IT", "LT", "LV", "MD",
    "ME", "NL", "NO", "PL", "PT", "RO", "RS",
    "SE", "SI", "SK"] 

# Names of sub varibales in files
GEN_VARS     = ["Wind", "Solar", "Hydro"]   # col of gen_data
LOAD_VAR     = "Load"                        # col of load_data
PRICE_VAR    = "Price"                       # col od price_data
RESERVE_VAR  = "RM"               # col of reserve_margin_data


EPU_FILE  = None   # es. Path("epu_daily.xlsx")
CO2_FILE  = None   # es. Path("co2_price.xlsx")

#load functions 

def load_excel(path: Path, date_col: int = 0) -> pd.DataFrame:
    df = pd.read_excel(path, index_col=date_col, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index()
    return df


def extract_columns(df: pd.DataFrame,
                    countries: list[str],
                    varname: str) -> pd.DataFrame:

    selected = {}
    for c in countries:
        col_name = f"{c}_{varname}"
        if col_name in df.columns:
            selected[col_name] = df[col_name]
        else:
            print(f"  ⚠ Col '{col_name}' not found in {df.columns.tolist()}")
    return pd.DataFrame(selected)

def impute(
    df: pd.DataFrame, 
    ffill_limit     : int = 2,
    spline_limit    : int = 7,
    seasonal_limit  : int = 16
    ) -> tuple[pd.DataFrame, dict]:
    
    """
    Gap 1-2 days  →  ffill      
    Gap 3-7 days  →  spline    
    Gap 8-16 days →  seasonal 
    Gap > 16 days →  NaN → report
    """
    
    
    df_imputed = df.copy()
    
    def gap_mask (s, max_len):
        is_nan = s.isna()
        block_id = (is_nan != is_nan.shift()).cumsum()
        sizes = is_nan.astype(int).groupby(block_id).transform('sum')
        return is_nan & (sizes <= max_len)
    
    df_imputed = df_imputed.ffill(limit = ffill_limit)
    
    for col in df_imputed.columns :
        mask = gap_mask (df_imputed[col], spline_limit)
        if not mask.any():
            continue
        observed = df_imputed[col].dropna()
        if len(observed) < 4:
            continue
        x_obs = observed.index.astype('int64')//10**9
        cs = CubicSpline(x_obs, observed.values, extrapolate = False)
        x_fill = df_imputed.index[mask].astype('int64')//10**9
        df_imputed.loc[mask, col] = cs(x_fill)
        
    for col in df_imputed.columns :
        mask = gap_mask (df_imputed[col], seasonal_limit)
        if not mask.any():
            continue
        dow_median = df_imputed[col].groupby(df_imputed.index.dayofweek).median()
        seasonal = pd.Series(
            df_imputed.index.dayofweek.map(dow_median).astype(float),
            index = df_imputed.index
        )
        residual = df_imputed[col] - seasonal
        res_filled = residual.interpolate(method='time', limit=seasonal_limit, limit_direction = 'forward')
        df_imputed.loc[mask, col] = (res_filled + seasonal).loc[mask]

    # reporting 
    report = {}
    for col in df_imputed.columns:
        nan_mask = df_imputed[col].isna()
        if nan_mask.any():
            # Groups consecutives NaN in blocks
            blocks = (nan_mask != nan_mask.shift()).cumsum()[nan_mask]
            block_lengths = blocks.value_counts().sort_index()
            report[col] = {
                "total_nan"    : int(nan_mask.sum()),
                "n_blocks"     : len(block_lengths),
                "max_block_len": int(block_lengths.max()),
                "dates"        : df_imputed.index[nan_mask].tolist()
            }

    if report:
        print("⚠ Residual NaN after ffill (missing > 2 days):")
        for col, info in report.items():
            print(f"  {col}: {info['total_nan']} NaN, "
                  f"{info['n_blocks']} block, "
                  f"max {info['max_block_len']} consecutive days")
    else:
        print("No resuidual Nan after ffill.")

    return df_imputed, report


# Aggregation and implemention
print("=" * 60)
print("LOADING FILES")
print("=" * 60)

df_gen     = load_excel(FILE_MAP["gen"])
df_load    = load_excel(FILE_MAP["load"])
df_price   = load_excel(FILE_MAP["price"])
df_reserve = load_excel(FILE_MAP["reserve"])

print(f"  gen_data     : {df_gen.shape}    col: {df_gen.columns.tolist()}")
print(f"  load_data    : {df_load.shape}   col: {df_load.columns.tolist()}")
print(f"  price_data   : {df_price.shape}  col: {df_price.columns.tolist()}")
print(f"  reserve_data : {df_reserve.shape} col: {df_reserve.columns.tolist()}")


print("\n" + "=" * 60)
print("EXCTRATION OD VARIABLES PER COUNTRY")
print("=" * 60)

# Endogenous vars
prices = extract_columns(df_price, COUNTRIES, PRICE_VAR)
loads  = extract_columns(df_load,  COUNTRIES, LOAD_VAR)

# Exogenous vars
winds  = extract_columns(df_gen, COUNTRIES, GEN_VARS[0])   # Wind
solars = extract_columns(df_gen, COUNTRIES, GEN_VARS[1])   # Solar
hydros = extract_columns(df_gen, COUNTRIES, GEN_VARS[2])   # Hydro
reserves = extract_columns(df_reserve, COUNTRIES, RESERVE_VAR)

#temporal intersection
all_indices = [
    df_gen.index, df_load.index, df_price.index, df_reserve.index
]

# Get progressive intersection
common_index = all_indices[0]
for idx in all_indices[1:]:
    common_index = common_index.intersection(idx)

print(f"\nSample: {common_index[0].date()} → {common_index[-1].date()}")
print(f"  T = {len(common_index)} daily observations")

# Build X and Y df

def align(df: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Filter a DataFrame to the common index"""
    return df.reindex(index)

# Build Y
Y = pd.concat([
    align(prices, common_index),
    align(loads,  common_index),
], axis=1)

# Build X
exo_parts = [
    align(winds,    common_index),
    align(solars,   common_index),
    align(hydros,   common_index),
    align(reserves, common_index),
]
X = pd.concat(exo_parts, axis=1)

#Quality check

print("\n" + "=" * 60)
print("QUALITY CHECK")
print("=" * 60)
print(f"\nY  — shape: {Y.shape}")
print(f"     NaN tot: {Y.isna().sum().sum()}")

print(f"\nX  — shape: {X.shape}")
print(f"     NaN tot: {X.isna().sum().sum()}")

Y, report_Y = impute(Y)
X, report_X = impute(X)

col_to_drop = ['IE_Solar', 'ME_Solar', 'NL_Hydro', 'MD_RM']
X = X.drop(columns = col_to_drop)
print(f"     NaN tot: {X.isna().sum().sum()}")

# Full dataset for inspection / debug
df_full = pd.concat([Y, X], axis=1)
df_full.to_excel(Path(__file__).parent / "df_full_check.xlsx")


# ARRAYS FOR GIBBS
"""
    T  = number of observations
    n  = number of endo vars  (= Y.shape[1])
    k  = number of exo vars   (= X.shape[1])
    p  = number of autoregressive lags   (TBD)
    q  = number of lag eco         (TBD)
"""

Y_np = Y.values.astype(float)    # [T × n]
X_np = X.values.astype(float)    # [T × k]
dates = Y.index                  

T, n = Y_np.shape
_, k = X_np.shape

print("\n" + "=" * 60)
print("ARRAY READY FOR GIBBS SAMPLER")
print("=" * 60)
print(f"  Y_np  : {Y_np.shape}   (T={T}, n={n} endo vars)")
print(f"  X_np  : {X_np.shape}   (T={T}, k={k} exo vars)")

np.save('data/Y.npy', Y_np)
np.save('data/X.npy', X_np)