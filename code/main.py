import pandas as pd
import networkx as nx

flows = pd.read_csv('data/physical_energy_power_flows_2025.csv', sep='\t')

exports = flows[flows['Direction'] == 'Export'][['FromCountryMapCode', 'ToCountryMapCode']].rename(
    columns={'FromCountryMapCode': 'source', 'ToCountryMapCode': 'target'}
)
imports = flows[flows['Direction'] == 'Import'][['ToCountryMapCode', 'FromCountryMapCode']].rename(
    columns={'ToCountryMapCode': 'source', 'FromCountryMapCode': 'target'}
)

links = pd.concat([exports, imports]).drop_duplicates()

G = nx.DiGraph()
nodes = sorted(list(set(links['source']) | set(links['target'])))
G.add_nodes_from(nodes)
G.add_edges_from(zip(links['source'], links['target']))

coords = {
    'AL': (41.3, 19.8), 'AM': (40.1, 44.5), 'AT': (47.5, 14.5), 'AZ': (40.2, 47.6),
    'BA': (44.2, 17.8), 'BE': (50.5, 4.5), 'BG': (42.7, 25.5), 'CH': (46.8, 8.2),
    'CZ': (49.8, 15.5), 'DE': (51.2, 10.4), 'DK': (56.0, 10.0), 'EE': (58.6, 25.0),
    'ES': (40.0, -3.7), 'FI': (62.0, 26.0), 'FR': (46.2, 2.2), 'GB': (54.0, -2.0),
    'GE': (42.0, 43.5), 'GR': (39.0, 22.0), 'HR': (45.1, 15.2), 'HU': (47.2, 19.5),
    'IE': (53.1, -7.7), 'IT': (41.9, 12.6), 'LT': (55.2, 23.9), 'LU': (49.8, 6.1),
    'LV': (56.9, 24.6), 'MD': (47.0, 28.9), 'ME': (42.7, 19.3), 'MK': (41.6, 21.7),
    'MT': (35.9, 14.4), 'NL': (52.2, 5.3), 'NO': (60.5, 8.5), 'PL': (51.9, 19.1),
    'PT': (39.4, -8.2), 'RO': (45.9, 25.0), 'RS': (44.0, 21.0), 'RU': (55.7, 37.6),
    'SE': (62.0, 15.0), 'SI': (46.1, 14.8), 'SK': (48.7, 19.1), 'TR': (39.0, 35.0),
    'UA': (48.4, 31.2), 'XK': (42.6, 21.1)
}

for node in G.nodes():
    if node in coords:
        lat, lon = coords[node]
        G.nodes[node]['latitude'] = float(lat)
        G.nodes[node]['longitude'] = float(lon)

print(f"Graph with {G.number_of_nodes()} nodes (countries) & {G.number_of_edges()} edges (physivcal connections).")
nx.write_gexf(G, 'graph_2025.gexf')
adj_matrix = nx.to_pandas_adjacency(G, dtype=int)