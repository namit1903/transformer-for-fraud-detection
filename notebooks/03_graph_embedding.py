from google.colab import drive
drive.mount('/content/drive')

# !pip install --upgrade numpy
import numpy as np
print(np.__version__)  # Version used when saving


# !pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html -q
# !pip install torch-geometric -q
# !pip install networkx -q


import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
import os

root = "/content/drive/MyDrive/FraudBehaviorEmbeddings"

cdr = pd.read_csv(os.path.join(root, "data/raw/Simulated_CDR_IPDR_Logs.csv"))
rtm = pd.read_csv(os.path.join(root, "data/raw/Simulated_RTM_Logs.csv"))
transformer_df = pd.read_pickle(os.path.join(root, "data/processed/transformer_input.pkl"))

# Create base graph
G = nx.Graph()

# Add subscriber nodes
subscribers = set(cdr['subscriber_id']) | set(rtm['subscriber_id'])
G.add_nodes_from(subscribers, node_type='user')

# Add app nodes and edges from RTM logs
for _, row in rtm.iterrows():
    user = row['subscriber_id']
    app = row['app']
    G.add_node(app, node_type='app')
    G.add_edge(user, app, duration=row['duration_sec'])

# Add destination domain nodes and edges from CDR logs
for _, row in cdr.iterrows():
    user = row['subscriber_id']
    domain = row['destination']
    G.add_node(domain, node_type='domain')
    G.add_edge(user, domain, duration=row['duration_sec'])

# Create enriched node features using transformer_input.pkl
# We'll use:
# - Length of behavior sequence
# - Number of unique actions
# - Node type one-hot

# import numpy as np

# Step 4.1: Map user behavior stats
behavior_map = {}
for _, row in transformer_df.iterrows():
    user = row['subscriber_id']
    actions = row['padded_sequence']
    behavior_map[user] = {
        "length": len(actions),
        "unique": len(set(actions))
    }

# Step 4.2: Assign feature vector to each node
for node in G.nodes():
    node_type = G.nodes[node]['node_type']

    # One-hot: [user, app, domain]
    one_hot = [int(node_type == 'user'), int(node_type == 'app'), int(node_type == 'domain')]

    if node_type == 'user':
        length = behavior_map.get(node, {}).get("length", 0)
        unique = behavior_map.get(node, {}).get("unique", 0)
        G.nodes[node]['x'] = [length, unique] + one_hot
    else:
        # For app/domain, just use dummy feature + one-hot
        G.nodes[node]['x'] = [1.0, 1.0] + one_hot

# Converting to PyG Data
from torch_geometric.utils import from_networkx

pyg_graph = from_networkx(G)
print(pyg_graph)

#GraphSAGE model
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GraphSAGEModel(in_channels=6, hidden_channels=32, out_channels=16)  # 6 = [length, unique, one-hot(3)]
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
def train():
    model.train()
    optimizer.zero_grad()
    out = model(pyg_graph.x.float(), pyg_graph.edge_index)
    loss = ((out - out.mean())**2).sum()  # dummy unsupervised loss
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Save embeddings
embeddings = model(pyg_graph.x.float(), pyg_graph.edge_index).detach().numpy()
np.save(os.path.join(root, "data/processed/graphsage_embeddings_enriched.npy"), embeddings)
print(" GraphSAGE embeddings saved!!!")

