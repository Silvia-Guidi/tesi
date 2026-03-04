import pandas as pd
import networkx as nx
df = pd.read_csv('physical_energy_power_flows_2025.csv', sep='\t')

exports = df[df['Direction'] == 'Export'][['FromCountryMapCode', 'ToCountryMapCode']].rename(
    columns={'FromCountryMapCode': 'source', 'ToCountryMapCode': 'target'}
)
imports = df[df['Direction'] == 'Import'][['ToCountryMapCode', 'FromCountryMapCode']].rename(
    columns={'ToCountryMapCode': 'source', 'FromCountryMapCode': 'target'}
)

connections = pd.concat([exports, imports]).drop_duplicates()

G = nx.DiGraph()

nodes = sorted(list(set(connections['source']) | set(connections['target'])))
G.add_nodes_from(nodes)

for _, row in connections.iterrows():
    G.add_edge(row['source'], row['target'])

print(f"Graph with {G.number_of_nodes()} nodes (countries) & {G.number_of_edges()} edges (physivcal connections).")