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
    parser = argparse.ArgumentParser(description="Costruisce matrici G da ENTSO-E Transfer Capacities")
    parser.add_argument("--data_dir",   type=str,   default=".",
                        help="Cartella contenente i 13 CSV ENTSO-E (default: cartella corrente)")
    parser.add_argument("--threshold",  type=float, default=500,
                        help="Soglia MW per binarizzare G₀ (default: 500 MW)")
    parser.add_argument("--agg",        type=str,   default="mean",
                        choices=["mean", "max", "median"],
                        help="Aggregazione annuale delle capacità MW (default: mean)")
    parser.add_argument("--output_dir", type=str,   default=".",
                        help="Cartella di output (default: cartella corrente)")
    return parser.parse_args()
 

 
def load_all_csvs(data_dir: str) -> pd.DataFrame:
    """
    Carica tutti i file CSV ENTSO-E ForecastedTransferCapacities
    e li concatena in un unico DataFrame.
    """
    pattern = os.path.join(data_dir, "*.csv")
    files   = sorted(glob.glob(pattern))
 
    if not files:
        raise FileNotFoundError(
            f"Nessun file CSV trovato in '{data_dir}'.\n"
            f"Controlla il parametro --data_dir."
        )
 
    print(f"  Trovati {len(files)} file CSV:")
    dfs = []
    for f in files:
        print(f"    • {os.path.basename(f)}")
        df = pd.read_csv(f, sep="\t", parse_dates=["DateTime(UTC)"])
        dfs.append(df)
 
    data = pd.concat(dfs, ignore_index=True)
    print(f"\n  Righe totali caricate: {len(data):,}")
    return data
 
 
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Pulizia: rinomina colonne, rimuove duplicati, filtra capacità nulle.
    """
    # Rinomina per comodità
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
 
    # Rimuovi duplicati (stesso timestamp + stessa coppia + stesso contratto)
    before = len(data)
    data = data.drop_duplicates(subset=["datetime", "from_code", "to_code", "contract_type"])
    print(f"  Duplicati rimossi: {before - len(data):,}")
 
    # Filtra capacità non-negativa e non-nulla
    data = data[data["capacity_mw"] > 0].copy()
    print(f"  Righe con capacità > 0: {len(data):,}")
 
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
    print(f"  Righe mantenute : {len(filtered):,} / {before:,}")
    if excluded:
        print(f"  MapCode esclusi : {', '.join(excluded)}")
    return filtered
 
 
def print_monthly_breakdown(data: pd.DataFrame):
    data["month"] = data["datetime"].dt.to_period("M")
    breakdown = data.groupby("month").size().sort_index()
    print("\n  Distribuzione mensile delle osservazioni:")
    for month, count in breakdown.items():
        print(f"    {month}: {count:,} righe")
    print(f"\n  → Aggregazione annuale su {len(breakdown)} mesi.\n"
          f"    G₀ sarà FISSA nel tempo: cattura la topologia media dell'anno.")
 
 

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
 
    print(f"  Paesi nella matrice ({n}): {all_nodes}")
    print(f"  Capacità max osservata: {cap_matrix.values.max():.0f} MW")
    print(f"  Capacità media (archi>0): {cap_df['capacity_mw'].mean():.0f} MW")
    return cap_matrix
 
 
def binarize(cap_matrix: pd.DataFrame, threshold: float) -> pd.DataFrame:
    G0 = (cap_matrix >= threshold).astype(int)
    np.fill_diagonal(G0.values, 0)  # Nessun auto-loop
    return G0
 

 
def main():
    args = parse_args()
 
    os.makedirs(args.output_dir, exist_ok=True)
    print("\n" + "=" * 60)
    print("=" * 60)
 
    # Caricamento
    print("Caricamento CSV...")
    data = load_all_csvs(args.data_dir)
 
    # Pulizia
    print("Pulizia dati...")
    data = clean_data(data)
 
    # Filtro paesi
    print("\n[3/6] Filtro paesi/zone...")
    data = filter_countries(data)
 
    # Matrice capacità
    print("Costruzione matrice capacità...")
    cap_matrix = build_capacity_matrix(data, args.agg)
 
    # Binarizzazione G0
    print(f"Binarizzazione con soglia {args.threshold} MW...")
    G0 = binarize(cap_matrix, args.threshold)
    
 
    # Salva CSV
    G0.to_csv(os.path.join(args.output_dir, "G0_matrix.csv"))
    cap_matrix.to_csv(os.path.join(args.output_dir, "G0_capacity_matrix.csv"))
    pd.Series(list(G0.index)).to_csv(
        os.path.join(args.output_dir, "zone_list.txt"),
        index=False, header=False
    )
    print(f"  G0_matrix.csv salvato ({G0.shape[0]}×{G0.shape[1]})")

 
 
if __name__ == "__main__":
    main()
 