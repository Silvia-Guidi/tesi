import os
import pandas as pd
import time
from entsoe import EntsoePandasClient
from tqdm import tqdm

# --- CONFIGURAZIONE ---
API_KEY = '4d659eb6-bacc-4b41-91a2-00b068ebbca2'
client = EntsoePandasClient(api_key=API_KEY)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Percorsi file esterni (Assicurati che siano nella cartella 'data')
EPU_FILE = os.path.join(DATA_DIR, "EPU.csv")

# Finestra Temporale
YEARS = [2023, 2024, 2025]

countries = [
    'AL', 'AT', 'BA', 'BE', 'BG', 'CH', 'CY', 'CZ', 'DE', 'DK', 
    'EE', 'ES', 'FI', 'FR', 'GE', 'GR', 'HR', 'HU', 'IE', 'IT', 
    'LT', 'LU', 'LV', 'ME', 'MK', 'NL', 'NO', 'PL', 'PT', 'RO', 
    'RS', 'SE', 'SI', 'SK', 'TR', 'UA', 'GB'
]

# --- 1. FUNZIONE DOWNLOAD ENTSO-E ---
def country_data(iso):
    file_path = os.path.join(DATA_DIR, f"{iso}_temp.csv")
    
    # Se il file esiste già, lo carichiamo (Checkpoint)
    if os.path.exists(file_path):
        return pd.read_csv(file_path, index_col=0, parse_dates=True)

    yearly_data = []
    for year in YEARS:
        start = pd.Timestamp(f'{year}0101', tz='UTC')
        end = pd.Timestamp(f'{year}1231', tz='UTC')
        
        # Tentativi multipli in caso di 503
        for attempt in range(3):
            try:
                l = client.query_load(iso, start=start, end=end)
                if isinstance(l, pd.DataFrame): l = l['Actual Load']
                
                p = client.query_day_ahead_prices(iso, start=start, end=end)
                g = client.query_generation(iso, start=start, end=end)
                
                # Creazione dataframe annuale
                df_year = pd.DataFrame({
                    f'{iso}_Price': p,
                    f'{iso}_Load': l,
                    f'{iso}_Solar': g.get('Solar', 0),
                    f'{iso}_Wind': g.get('Wind Onshore', 0) + g.get('Wind Offshore', 0),
                    f'{iso}_Hydro': g.get('Hydro Run-of-river and poundage', 0)
                })
                yearly_data.append(df_year)
                time.sleep(2) # Pausa tra anni
                break 
            except Exception as e:
                print(f"Tentativo {attempt+1} fallito per {iso} ({year}): {e}")
                time.sleep(10) # Pausa più lunga dopo errore
    
    if yearly_data:
        full_country_df = pd.concat(yearly_data).resample('h').mean()
        full_country_df.to_csv(file_path) # Salvataggio checkpoint
        return full_country_df
    return None


# --- 2. ELABORAZIONE EPU (Mensile -> Orario) ---
def get_epu_mapped(target_index):
    if not os.path.exists(EPU_FILE):
        print("ATTENZIONE: File EPU non trovato. Salto integrazione EPU.")
        return pd.DataFrame(index=target_index)
    
    df_epu = pd.read_csv(EPU_FILE).dropna(subset=['Year', 'Month'])
    df_epu['Timestamp'] = pd.to_datetime(df_epu[['Year', 'Month']].assign(Day=1)).dt.tz_localize('UTC')
    df_epu = df_epu.set_index('Timestamp')
    
    # Mapping specifico per paese
    mapping = {'IT': 'Italy_News_Index', 'DE': 'Germany_News_Index', 
               'FR': 'France_News_Index', 'ES': 'Spain_News_Index', 'GB': 'UK_News_Index'}
    
    epu_final = pd.DataFrame(index=target_index)
    for iso in countries:
        source_col = mapping.get(iso, 'European_News_Index')
        epu_final[f'{iso}_EPU'] = df_epu[source_col].reindex(target_index, method='ffill')
    return epu_final


# --- ESECUZIONE MAIN ---

# Download ENTSO-E
entsoe_list = []
# Ciclo principale
all_dfs = []
for iso in tqdm(countries, desc="Download Paesi"):
    df = country_data(iso)
    if df is not None:
        all_dfs.append(df)

if entsoe_list:
    dataset = pd.concat(entsoe_list, axis=1)
    dataset = dataset.interpolate(method='linear', limit=3).ffill().fillna(0)
    
    # Integrazione EPU 
    print("Integrazione variabili macroeconomiche...")
    df_epu_h = get_epu_mapped(dataset.index)
    
    final_df = pd.concat([dataset, df_epu_h], axis=1)
    
    # Salvataggio
    final_df.to_csv(os.path.join(DATA_DIR, "dataset_2023_2025.csv"))
    print(f"\nDataset finale pronto! Formato: {final_df.shape}")