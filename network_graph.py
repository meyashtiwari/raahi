import os
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from modules.coordinates_generator import coordinates_generator

file_path = os.path.join('data', 'test_dataset.csv')
data_nodes = pd.read_csv(file_path)

nodes = data_nodes[['Class', 'Range', 'Network_type']]

random_nodes = nodes.sample(n=10)
random_nodes = random_nodes.reset_index()
random_nodes.rename(columns={'index': 'Node_id'}, inplace=True)

cg = coordinates_generator(10)
coordinates = pd.DataFrame(cg.generateRandomPoints())

nodes_data = pd.concat([random_nodes, coordinates], axis=1)

print(nodes_data)

# G = nx.Graph()

# G.add_edge("a", "b", weight=0.6)
# G.add_edge("a", "c", weight=0.8)
# G.add_edge("c", "d", weight=0.1)
# G.add_edge("c", "e", weight=0.7)
# G.add_edge("c", "f", weight=0.9)
# G.add_edge("a", "d", weight=0.3)

# elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
# esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

# pos = nx.spring_layout(G, seed=4)  # positions for all nodes - seed for reproducibility

# # nodes
# nx.draw_networkx_nodes(G, pos, node_size=600)

# # edges
# nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
# nx.draw_networkx_edges(
#     G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
# )

# # labels
# nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

# ax = plt.gca()
# ax.margins(0.08)
# plt.axis("off")
# plt.tight_layout()
# plt.show()