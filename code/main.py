from entsoe import EntsoePandasClient
from entsoe.mappings import NEIGHBOURS
import pandas as pd
import numpy as np
import networkx as nx
import time

API_KEY = 'IL_TUO_TOKEN_QUI'
client = EntsoePandasClient(api_key=API_KEY)

countries = [
    'AL', 'AT', 'BA', 'BE', 'BG', 'CH', 'CY', 'CZ', 'DE', 'DK', 
    'EE', 'ES', 'FI', 'FR', 'GE', 'GR', 'HR', 'HU', 'IE', 'IT', 
    'LT', 'LU', 'LV', 'ME', 'MK', 'NL', 'NO', 'PL', 'PT', 'RO', 
    'RS', 'SE', 'SI', 'SK', 'TR', 'UA', 'UK'
]

start = pd.Timestamp('20180101', tz='UTC')
end = pd.Timestamp('20260301', tz='UTC') 

# Network
G = nx.DiGraph()
G.add_nodes_from(countries)

for node_from, neighbors in NEIGHBOURS.items():
    clean_from = node_from.split('_')[0]
    
    if clean_from in countries:
        for node_to in neighbors:
            clean_to = node_to.split('_')[0]
            
            if clean_to in countries and clean_to != clean_from:
                G.add_edge(clean_from, clean_to)

print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_nodes()}")
adj_matrix = nx.to_pandas_adjacency(G, nodelist=countries, dtype=int)
print(adj_matrix.head())

# Var
queries = {
    'Load': client.query_load,             
    'Gen': client.query_generation     
    }   

data_init = {}

for iso in countries:
    try:
        print(f"--- Downloading all variables for: {iso} ---")
        
        l = queries['Load'](iso, start=start, end=end)
        if isinstance(l, pd.DataFrame): l = l['Actual Load']
        l = l.resample('H').mean().clip(lower=0)
        g = queries['Gen'](iso, start=start, end=end).resample('H').mean()
        
        g_f = client.query_generation_forecast(iso, start=start, end=end).resample('H').mean()
        if isinstance(g_f, pd.DataFrame): g_f = g_f.iloc[:, 0]
        l_f = client.query_load_forecast(iso, start=start, end=end).resample('H').mean()
        if isinstance(l_f, pd.DataFrame): l_f = l_f.iloc[:, 0]
        reserve_margin = ((g_f - l_f) / l_f)
    
        df_iso = pd.DataFrame({
            f'{iso}_Load_Actual': l, 
            f'{iso}_Wind': g.get('Wind Onshore', 0)+ g.get('Wind Offshore', 0).clip(lower=0), 
            f'{iso}_Solar': g.get('Solar', 0).clip(lower=0), 
            f'{iso}_Hydro': g.get('Hydro Run-of-river and poundage', 0).clip(lower=0) ,
            f'{iso}_Reserve_Margin': reserve_margin
        })
        
        data_init[iso] = df_iso
        time.sleep(1) 
        
    except Exception as e:
        print(f"Errore su {iso}: {e}")


data_raw = pd.concat(data_init.values(), axis=1)
def clean_data(data):
    data = data.interpolate(method='linear', limit=3)
    
    for col in data.columns:
        mean = data[col].mean()
        std = data[col].std()
        data[col] = data[col].clip(upper=mean + 6 * std)
    return data

data= clean_data(data_raw)