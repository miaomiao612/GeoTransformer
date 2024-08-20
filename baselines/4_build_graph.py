import pandas as pd
import json
import networkx as nx
import numpy as np
import pickle

grid_info = pd.read_csv('latent_space.csv', index_col=0)

with open('center_distances.json', 'r') as f:
    distances = json.load(f)

# with open('GDP_results.json', 'r') as f:
#     data = json.load(f)

with open('Trips_results.json', 'r') as f:
    trip_data = json.load(f)

distance_threshold = 0.3073

G = nx.Graph()

for grid_id, features in grid_info.iterrows():
    G.add_node(grid_id, features=features.tolist(), trip=trip_data[grid_id]["Trips"])

for edge, distance in distances.items():
    if distance <= distance_threshold:
        node1, node2 = edge.split('-')
        node1 += '.tif'
        node2 += '.tif'
        if node1 in G and node2 in G:
            G.add_edge(node1, node2, weight=1)

with open("graph_trips.pkl", "wb") as f:
    pickle.dump(G, f)

# Graph Info
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
