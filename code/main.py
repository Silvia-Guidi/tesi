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

print(f"Graph with {G.number_of_nodes()} nodes (countries) & {G.number_of_edges()} edges (physivcal connections).")
nx.write_gexf(G, 'graph_2025.gexf')
adj_matrix = nx.to_pandas_adjacency(G, dtype=int)