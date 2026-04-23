"""
INPUT FILES (xlsx):
    gen_data            -> Renewable generation (wind, solar, hydro) per country
    load_data           -> Actual total load per country
    price_data          -> Electricity prices (intraday or day-ahead) per country
    reserve_margin_data -> Reserve margin per country

COLUMN CONVENTION in each file:
    {COUNTRY}_{variable}  e.g., "IT_Wind", "DE_Solar", "FR_Load", "IT_Price"

OUTPUT:
    Y.npy  -> array [T x n_endo]  interleaved layout: [p_AT, d_AT, p_BE, d_BE, ...]
    X.npy  -> array [T x n_exo]   exogenous block (wind, solar, hydro, reserve margin)
    countries_final.csv  -> ordered list of countries actually used
    df_full_check.xlsx   -> full merged dataframe for inspection
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline

DATA_DIR = Path(__file__).parent

FILE_MAP = {
    "gen"     : DATA_DIR / "gen_data"              / "energy_pivot.xlsx",
    "load"    : DATA_DIR / "load_data"             / "load_pivot.xlsx",
    "price"   : DATA_DIR / "price_data"            / "prices_pivot.xlsx",
    "reserve" : DATA_DIR / "reserve_margin_data"   / "reserve_margin_data.xlsx",
}

# Countries to consider (the filtering step will keep only those with BOTH Price and Load)
COUNTRIES = ["AT", "BA", "BE", "BG", "CH", "CZ", "DE",
    "DK", "EE", "ES", "FI", "FR",
    "GR", "HR", "HU", "IE", "IT", "LT", "LV", "MD",
    "ME", "NL", "NO", "PL", "PT", "RO", "RS",
    "SE", "SI", "SK"]

# Sub-variable names expected in each file
GEN_VARS    = ["Wind", "Solar", "Hydro"]
LOAD_VAR    = "Load"
PRICE_VAR   = "Price"
RESERVE_VAR = "RM"


# ---------------------------------------------------------------
# LOADING HELPERS
# ---------------------------------------------------------------
def load_excel(path: Path, date_col: int = 0) -> pd.DataFrame:
    df = pd.read_excel(path, index_col=date_col, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index()
    return df


def extract_columns(df: pd.DataFrame, countries: list, varname: str) -> pd.DataFrame:
    """Keep only columns of the form {COUNTRY}_{varname}; warn on missing ones."""
    selected = {}
    for c in countries:
        col_name = f"{c}_{varname}"
        if col_name in df.columns:
            selected[col_name] = df[col_name]
        else:
            print(f"  [!] Col '{col_name}' not found")
    return pd.DataFrame(selected)


# ---------------------------------------------------------------
# IMPUTATION
# ---------------------------------------------------------------
def impute(
    df: pd.DataFrame,
    ffill_limit    : int = 2,
    spline_limit   : int = 7,
    seasonal_limit : int = 16,
) -> tuple[pd.DataFrame, dict]:
    """
    Gap 1-2 days  -> ffill
    Gap 3-7 days  -> spline
    Gap 8-16 days -> seasonal (day-of-week median)
    Gap > 16 days -> stays NaN, reported
    """
    df_imputed = df.copy()

    def gap_mask(s, max_len):
        is_nan   = s.isna()
        block_id = (is_nan != is_nan.shift()).cumsum()
        sizes    = is_nan.astype(int).groupby(block_id).transform("sum")
        return is_nan & (sizes <= max_len)

    df_imputed = df_imputed.ffill(limit=ffill_limit)

    for col in df_imputed.columns:
        mask = gap_mask(df_imputed[col], spline_limit)
        if not mask.any():
            continue
        observed = df_imputed[col].dropna()
        if len(observed) < 4:
            continue
        x_obs  = observed.index.astype("int64") // 10**9
        cs     = CubicSpline(x_obs, observed.values, extrapolate=False)
        x_fill = df_imputed.index[mask].astype("int64") // 10**9
        df_imputed.loc[mask, col] = cs(x_fill)

    for col in df_imputed.columns:
        mask = gap_mask(df_imputed[col], seasonal_limit)
        if not mask.any():
            continue
        dow_median = df_imputed[col].groupby(df_imputed.index.dayofweek).median()
        seasonal = pd.Series(
            df_imputed.index.dayofweek.map(dow_median).astype(float),
            index=df_imputed.index,
        )
        residual   = df_imputed[col] - seasonal
        res_filled = residual.interpolate(
            method="time", limit=seasonal_limit, limit_direction="forward"
        )
        df_imputed.loc[mask, col] = (res_filled + seasonal).loc[mask]

    # Report residual NaN
    report = {}
    for col in df_imputed.columns:
        nan_mask = df_imputed[col].isna()
        if nan_mask.any():
            blocks        = (nan_mask != nan_mask.shift()).cumsum()[nan_mask]
            block_lengths = blocks.value_counts().sort_index()
            report[col] = {
                "total_nan"    : int(nan_mask.sum()),
                "n_blocks"     : len(block_lengths),
                "max_block_len": int(block_lengths.max()),
                "dates"        : df_imputed.index[nan_mask].tolist(),
            }

    if report:
        print("[!] Residual NaN after imputation (gap > 16 days):")
        for col, info in report.items():
            print(f"  {col}: {info['total_nan']} NaN, "
                  f"{info['n_blocks']} block(s), "
                  f"max {info['max_block_len']} consecutive days")
    else:
        print("No residual NaN after imputation.")

    return df_imputed, report


# ---------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------
print("=" * 60)
print("LOADING FILES")
print("=" * 60)

df_gen     = load_excel(FILE_MAP["gen"])
df_load    = load_excel(FILE_MAP["load"])
df_price   = load_excel(FILE_MAP["price"])
df_reserve = load_excel(FILE_MAP["reserve"])

print(f"  gen_data     : {df_gen.shape}")
print(f"  load_data    : {df_load.shape}")
print(f"  price_data   : {df_price.shape}")
print(f"  reserve_data : {df_reserve.shape}")


# ---------------------------------------------------------------
# COUNTRY FILTERING (balanced panel required by BGVAR)
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("COUNTRY FILTERING (balanced panel required by BGVAR)")
print("=" * 60)

# Try to extract Price and Load for all COUNTRIES (some will be missing)
prices_full = extract_columns(df_price, COUNTRIES, PRICE_VAR)
loads_full  = extract_columns(df_load,  COUNTRIES, LOAD_VAR)

countries_with_price = {c for c in COUNTRIES if f"{c}_{PRICE_VAR}" in prices_full.columns}
countries_with_load  = {c for c in COUNTRIES if f"{c}_{LOAD_VAR}"  in loads_full.columns}

# Keep only countries with BOTH endogenous series, preserving COUNTRIES order
INCLUDED = [c for c in COUNTRIES if c in countries_with_price and c in countries_with_load]
EXCLUDED = [c for c in COUNTRIES if c not in INCLUDED]

print(f"Initial COUNTRIES  : {len(COUNTRIES)}")
print(f"  with Price       : {len(countries_with_price)}")
print(f"  with Load        : {len(countries_with_load)}")
print(f"  with BOTH (kept) : {len(INCLUDED)}  -> ny = {2 * len(INCLUDED)}")
print(f"  dropped          : {len(EXCLUDED)}")
print(f"\nIncluded ({len(INCLUDED)}):")
print(" ", ", ".join(INCLUDED))

if EXCLUDED:
    print(f"\nExcluded ({len(EXCLUDED)}):")
    for c in EXCLUDED:
        miss = []
        if c not in countries_with_price: miss.append("Price")
        if c not in countries_with_load:  miss.append("Load")
        print(f"  {c}: missing {', '.join(miss)}")

# Persist the final country list for downstream G0_matrix filtering
pd.Series(INCLUDED, name="country").to_csv(
    DATA_DIR / "countries_final.csv", index=False
)
print(f"\nSaved: {DATA_DIR / 'countries_final.csv'}")


# ---------------------------------------------------------------
# RE-EXTRACTION FOR INCLUDED COUNTRIES ONLY
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("EXTRACTION OF VARIABLES PER COUNTRY (INCLUDED ONLY)")
print("=" * 60)

prices   = extract_columns(df_price,   INCLUDED, PRICE_VAR)
loads    = extract_columns(df_load,    INCLUDED, LOAD_VAR)
winds    = extract_columns(df_gen,     INCLUDED, GEN_VARS[0])
solars   = extract_columns(df_gen,     INCLUDED, GEN_VARS[1])
hydros   = extract_columns(df_gen,     INCLUDED, GEN_VARS[2])
reserves = extract_columns(df_reserve, INCLUDED, RESERVE_VAR)


# ---------------------------------------------------------------
# TEMPORAL INTERSECTION
# ---------------------------------------------------------------
all_indices = [df_gen.index, df_load.index, df_price.index, df_reserve.index]
common_index = all_indices[0]
for idx in all_indices[1:]:
    common_index = common_index.intersection(idx)

print(f"\nSample: {common_index[0].date()} -> {common_index[-1].date()}")
print(f"  T = {len(common_index)} daily observations")


# ---------------------------------------------------------------
# BUILD Y (INTERLEAVED) AND X
# ---------------------------------------------------------------
# Align to common index
prices = prices.reindex(common_index)
loads  = loads.reindex(common_index)

# Y in INTERLEAVED layout: [p_AT, d_AT, p_BE, d_BE, ...]
# This matches expand_G0's assumption of 2 contiguous variables per country.
Y_cols = []
for c in INCLUDED:
    Y_cols.append(prices[f"{c}_{PRICE_VAR}"])
    Y_cols.append(loads [f"{c}_{LOAD_VAR}" ])
Y = pd.concat(Y_cols, axis=1)

# X (exogenous block - column order not critical for the model)
exo_parts = [
    winds.reindex(common_index),
    solars.reindex(common_index),
    hydros.reindex(common_index),
    reserves.reindex(common_index),
]
X = pd.concat(exo_parts, axis=1)


# ---------------------------------------------------------------
# QUALITY CHECK AND IMPUTATION
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("QUALITY CHECK AND IMPUTATION")
print("=" * 60)

print(f"\nY shape before imputation: {Y.shape}, NaN total: {Y.isna().sum().sum()}")
Y, report_Y = impute(Y)

print(f"\nX shape before imputation: {X.shape}, NaN total: {X.isna().sum().sum()}")
X, report_X = impute(X)

# Drop columns that are unreliable (hard-coded exclusions)
col_to_drop = [c for c in ["IE_Solar", "ME_Solar", "NL_Hydro", "MD_RM"] if c in X.columns]
if col_to_drop:
    X = X.drop(columns=col_to_drop)
    print(f"\nDropped from X: {col_to_drop}")
print(f"X shape after cleanup  : {X.shape}, NaN total: {X.isna().sum().sum()}")


# ---------------------------------------------------------------
# SAVE FULL DATASET FOR INSPECTION
# ---------------------------------------------------------------
df_full = pd.concat([Y, X], axis=1)
df_full.to_excel(DATA_DIR / "df_full_check.xlsx")


# ---------------------------------------------------------------
# ARRAYS FOR GIBBS SAMPLER
# ---------------------------------------------------------------
Y_np  = Y.values.astype(float)    # [T x n_endo]
X_np  = X.values.astype(float)    # [T x n_exo]
dates = Y.index

T, n = Y_np.shape
_, k = X_np.shape


# ---------------------------------------------------------------
# STANDARDIZATION (column-wise: zero mean, unit variance)
# ---------------------------------------------------------------
# Essential because price (EUR/MWh, O(1e2)) and load (MW, O(1e4)) live on
# very different scales. A raw Y.T @ Y is badly conditioned and breaks
# Cholesky / slogdet / Wishart sampling inside the Gibbs sampler.
# We save the scaling factors so posterior results can be back-transformed
# to the original units (EUR/MWh, MW) for interpretation.

print("\n" + "=" * 60)
print("STANDARDIZATION (zero mean, unit variance per column)")
print("=" * 60)

# Endogenous variables
y_mean    = Y_np.mean(axis=0)
y_std_dev = Y_np.std(axis=0, ddof=1)

if np.any(y_std_dev < 1e-10):
    bad_cols = [Y.columns[i] for i in np.where(y_std_dev < 1e-10)[0]]
    raise ValueError(f"Zero-variance columns in Y: {bad_cols}")

Y_std = (Y_np - y_mean) / y_std_dev

# Exogenous variables
x_mean    = X_np.mean(axis=0)
x_std_dev = X_np.std(axis=0, ddof=1)

if np.any(x_std_dev < 1e-10):
    bad_cols = [X.columns[i] for i in np.where(x_std_dev < 1e-10)[0]]
    print(f"  [!] Zero-variance columns in X (will be kept at 0): {bad_cols}")
    x_std_dev = np.where(x_std_dev < 1e-10, 1.0, x_std_dev)  # avoid div-by-zero

X_std = (X_np - x_mean) / x_std_dev

print(f"Y raw      : min={Y_np.min():>10.2f}, max={Y_np.max():>10.2f}")
print(f"Y standard : min={Y_std.min():>10.2f}, max={Y_std.max():>10.2f}  "
      f"(mean~0, std~1)")
print(f"X raw      : min={X_np.min():>10.2f}, max={X_np.max():>10.2f}")
print(f"X standard : min={X_std.min():>10.2f}, max={X_std.max():>10.2f}  "
      f"(mean~0, std~1)")


# ---------------------------------------------------------------
# SAVE EVERYTHING
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("ARRAYS READY FOR GIBBS SAMPLER")
print("=" * 60)
print(f"  Y_std  : {Y_std.shape}   (T={T}, n={n} endo vars = 2 x {n // 2} countries)")
print(f"  X_std  : {X_std.shape}   (T={T}, k={k} exo vars)")
print(f"  Layout of Y columns (interleaved):")
for i, col in enumerate(Y.columns):
    print(f"    {i:>3}: {col}")

# Save standardized arrays (what the Gibbs sampler will read)
np.save(DATA_DIR / "Y.npy", Y_std)
np.save(DATA_DIR / "X.npy", X_std)

# Save scaling factors for back-transformation of posterior results
np.save(DATA_DIR / "Y_mean.npy",    y_mean)
np.save(DATA_DIR / "Y_std_dev.npy", y_std_dev)
np.save(DATA_DIR / "X_mean.npy",    x_mean)
np.save(DATA_DIR / "X_std_dev.npy", x_std_dev)

print(f"\nSaved: {DATA_DIR / 'Y.npy'}        (standardized)")
print(f"Saved: {DATA_DIR / 'X.npy'}        (standardized)")
print(f"Saved: {DATA_DIR / 'Y_mean.npy'}   (for back-transformation)")
print(f"Saved: {DATA_DIR / 'Y_std_dev.npy'}")
print(f"Saved: {DATA_DIR / 'X_mean.npy'}")
print(f"Saved: {DATA_DIR / 'X_std_dev.npy'}")