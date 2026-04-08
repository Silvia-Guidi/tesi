import pandas as pd
import numpy as np
import os
import glob
import argparse
from pathlib import Path
 
 
 
ALLOWED_AREA_CODES = {
    "AT", "BA", "BE", "BG", "CH", "CY", "CZ", "DE",
    "DK", "EE", "ES", "FI", "FR", "GB", "GE",
    "GR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MD",
    "ME", "NL", "NO", "PL", "PT", "RO", "RS",
    "SE", "SI", "SK"
}  
 
 
 
def parse_args():
    parser = argparse.ArgumentParser(description="Build Matric G from ENTSO-E Transfer Capacities")
    parser.add_argument("--data_dir",   type=str,   default=".",
                        help="Path with  13 CSV ENTSO-E")
    parser.add_argument("--threshold",  type=float, default=500,
                        help="Threshold MW for binarization G₀ (default: 500 MW)")
    parser.add_argument("--agg",        type=str,   default="mean",
                        choices=["mean", "max", "median"],
                        help="Annual aggregation of capacity MW (default: mean)")
    parser.add_argument("--output_dir", type=str,   default=".",
                        help="Path of output")
    return parser.parse_args()
 

 
def load_all_csvs(data_dir: str) -> pd.DataFrame:
    pattern = os.path.join(data_dir, "*.csv")
    files   = sorted(glob.glob(pattern))
 
    if not files:
        raise FileNotFoundError(
            f"no file CSV found in '{data_dir}'.\n"
            f"check param --data_dir."
        )
 
    print(f"  found {len(files)} CSV files:")
    dfs = []
    for f in files:
        print(f"    • {os.path.basename(f)}")
        df = pd.read_csv(f, sep="\t", parse_dates=["DateTime(UTC)"])
        dfs.append(df)
 
    data = pd.concat(dfs, ignore_index=True)
    print(f"\n  Total rows loaded: {len(data):,}")
    return data
 
 
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    # Rename
    data = data.rename(columns={
        "DateTime(UTC)"              : "datetime",
        "OutAreaCode"                : "from_code",
        "OutAreaDisplayName"         : "from_name",
        "OutMapCode"                 : "from_map",
        "InAreaCode"                 : "to_code",
        "InAreaDisplayName"          : "to_name",
        "InMapCode"                  : "to_map",
        "ForecastTransferCapacity[MW]": "capacity_mw",
        "ContractType"               : "contract_type",
        "ResolutionCode"             : "resolution",
    })
    
    data["from_map"] = data["from_map"].apply(lambda x: "DE" if str(x).startswith("DE_") else x)
    data["to_map"] = data["to_map"].apply(lambda x: "DE" if str(x).startswith("DE_") else x)
 
    # Remove duplicates
    before = len(data)
    data = data.drop_duplicates(subset=["datetime", "from_code", "to_code", "contract_type"])
    print(f"  Duplicati rimossi: {before - len(data):,}")
 
    # Filer non-neg and non-null capacity 
    data = data[data["capacity_mw"] > 0].copy()
    print(f"  Rows with caacity > 0: {len(data):,}")
 
    return data
 
 

def filter_countries(data: pd.DataFrame) -> pd.DataFrame:
    before = len(data)
    mask = (
        data["from_map"].isin(ALLOWED_AREA_CODES) &
        data["to_map"].isin(ALLOWED_AREA_CODES)
    )
    filtered = data[mask].copy()
    excluded = sorted(
        (set(data["from_map"].unique()) | set(data["to_map"].unique())) - ALLOWED_AREA_CODES
    )
    print(f"  Rows kept : {len(filtered):,} / {before:,}")
    if excluded:
        print(f"  MapCode excluded : {', '.join(excluded)}")
    return filtered
 
 
def print_monthly_breakdown(data: pd.DataFrame):
    data["month"] = data["datetime"].dt.to_period("M")
    breakdown = data.groupby("month").size().sort_index()
    print("\n  Monthly distribution:")
    for month, count in breakdown.items():
        print(f"    {month}: {count:,} righe")
    print(f"\n  → Anjual aggregation on {len(breakdown)} mesi.\n")
 
 

def build_capacity_matrix(data: pd.DataFrame, agg: str) -> pd.DataFrame:
    data["from_node"] = data["from_map"]
    data["to_node"]   = data["to_map"]
 
    if agg == "mean":
        cap = data.groupby(["from_node", "to_node"])["capacity_mw"].mean()
    elif agg == "max":
        cap = data.groupby(["from_node", "to_node"])["capacity_mw"].max()
    elif agg == "median":
        cap = data.groupby(["from_node", "to_node"])["capacity_mw"].median()
 
    cap_df    = cap.reset_index()
    all_nodes = sorted(set(cap_df["from_node"]) | set(cap_df["to_node"]))
    n = len(all_nodes)
 
    cap_matrix = pd.DataFrame(0.0, index=all_nodes, columns=all_nodes)
    for _, row in cap_df.iterrows():
        cap_matrix.loc[row["from_node"], row["to_node"]] = row["capacity_mw"]
 
    print(f"  Countries in matrix ({n}): {all_nodes}")
    print(f"  Max capacity observed: {cap_matrix.values.max():.0f} MW")
    print(f"  Mean capacity (edges>0): {cap_df['capacity_mw'].mean():.0f} MW")
    return cap_matrix
 
 
def binarize(cap_matrix: pd.DataFrame, threshold: float) -> pd.DataFrame:
    G0 = (cap_matrix >= threshold).astype(int)
    np.fill_diagonal(G0.values, 0)  # No auto-loop
    return G0
 

 
def main():
    args = parse_args()
 
    os.makedirs(args.output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("=" * 60)
 
    # Load
    print("Loading CSV...")
    data = load_all_csvs(args.data_dir)
 
    # cleaning
    print("Cleaning data...")
    data = clean_data(data)
 
    # Country filter
    print("Filtering country...")
    data = filter_countries(data)
 
    # Capacity matrix
    print("Building capacity matrix...")
    cap_matrix = build_capacity_matrix(data, args.agg)
 
    # Binarize G0
    print(f"Binarize with threshold {args.threshold} MW...")
    G0 = binarize(cap_matrix, args.threshold)
    
 
    # Save CSV
    G0.to_csv(os.path.join(args.output_dir, "G0_matrix.csv"))
    cap_matrix.to_csv(os.path.join(args.output_dir, "G0_capacity_matrix.csv"))
    pd.Series(list(G0.index)).to_csv(
        os.path.join(args.output_dir, "zone_list.txt"),
        index=False, header=False
    )
    print(f"  G0_matrix.csv saved ({G0.shape[0]}×{G0.shape[1]})")

 
 
if __name__ == "__main__":
    main()
 