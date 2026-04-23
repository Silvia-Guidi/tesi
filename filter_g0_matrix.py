"""
Filter G0_matrix.csv to keep only the countries that ended up in Y.npy.

Reads:
  data/network_data/G0_matrix.csv   (original: 31 countries)
  data/countries_final.csv           (produced by data_ready.py, 28 countries)

Writes:
  data/network_data/G0_matrix.csv   (overwritten: 28 x 28)
  data/network_data/G0_matrix_original_backup.csv  (safety copy)

Run after data_ready.py.
"""

from pathlib import Path
import pandas as pd

DATA_DIR    = Path("data")
NETWORK_DIR = DATA_DIR / "network_data"

G0_PATH        = NETWORK_DIR / "G0_matrix.csv"
BACKUP_PATH    = NETWORK_DIR / "G0_matrix_original_backup.csv"
COUNTRIES_PATH = DATA_DIR / "countries_final.csv"

# --- Load the adjacency matrix and the final country list ---
G0 = pd.read_csv(G0_PATH, index_col=0)
countries = pd.read_csv(COUNTRIES_PATH)["country"].tolist()

print("=" * 60)
print("G0_MATRIX FILTERING")
print("=" * 60)
print(f"Original G0        : {G0.shape[0]} x {G0.shape[1]}")
print(f"  rows   : {list(G0.index)}")
print(f"  cols   : {list(G0.columns)}")
print(f"Countries to keep  : {len(countries)}")
print(f"  list   : {countries}")

# --- Sanity check: are all 'countries' actually present in G0? ---
missing_in_G0 = [c for c in countries if c not in G0.index]
if missing_in_G0:
    raise ValueError(
        f"These countries are in countries_final.csv but NOT in G0_matrix.csv: "
        f"{missing_in_G0}. Update G0_matrix.csv or remove them."
    )

# --- Keep backup of the original (only if not already present) ---
if not BACKUP_PATH.exists():
    G0.to_csv(BACKUP_PATH)
    print(f"\nBackup saved: {BACKUP_PATH}")

# --- Filter to the included countries, preserving the countries_final.csv order ---
G0_filtered = G0.loc[countries, countries]
print(f"\nFiltered G0        : {G0_filtered.shape[0]} x {G0_filtered.shape[1]}")
print(f"Active arcs        : {int(G0_filtered.values.sum())}")

# --- Dropped countries report ---
dropped = [c for c in G0.index if c not in countries]
print(f"\nDropped countries  : {dropped}")

# --- Save ---
G0_filtered.to_csv(G0_PATH)
print(f"\nSaved (overwritten): {G0_PATH}")
