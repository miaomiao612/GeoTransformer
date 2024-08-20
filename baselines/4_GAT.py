import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import networkx as nx
import pickle
import torch.optim as optim
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

with open("graph_trips.pkl", "rb") as f:
    G = pickle.load(f)

data = from_networkx(G)

data.x = torch.tensor([G.nodes[n]['features'] for n in G.nodes()], dtype=torch.float)
data.y = torch.tensor([G.nodes[n]['trip'] for n in G.nodes()], dtype=torch.float).unsqueeze(-1)

num_nodes = data.num_nodes
train_mask = torch.zeros(num_nodes, dtype=torch.bool)
test_mask = torch.zeros(num_nodes, dtype=torch.bool)

train_indices = np.random.choice(num_nodes, int(0.8 * num_nodes), replace=False)
test_indices = np.setdiff1d(np.arange(num_nodes), train_indices)

train_mask[train_indices] = True
test_mask[test_indices] = True

data.train_mask = train_mask
data.test_mask = test_mask

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        
        self.convs.append(GATConv(in_channels, hidden_channels, heads=8, dropout=0.1))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * 8, hidden_channels, heads=8, dropout=0.1))
        
        self.convs.append(GATConv(hidden_channels * 8, out_channels, heads=1, concat=False, dropout=0.1))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

model = GAT(in_channels=data.num_features, hidden_channels=8, out_channels=1, num_layers=4)


criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)



train_loader = NeighborLoader(data, num_neighbors=[4, 4], batch_size=16, shuffle=True, input_nodes=data.train_mask)
test_loader = NeighborLoader(data, num_neighbors=[4, 4], batch_size=16, shuffle=False, input_nodes=data.test_mask)


for epoch in range(150):
    model.train()
    train_loss = 0
    train_outputs = []
    train_targets = []
    
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_outputs.append(out[batch.train_mask].detach().cpu().numpy())
        train_targets.append(batch.y[batch.train_mask].detach().cpu().numpy())
    
    avg_train_loss = train_loss / len(train_loader)
    train_outputs = np.concatenate(train_outputs, axis=0)
    train_targets = np.concatenate(train_targets, axis=0)
    train_r2 = r2_score(train_targets, train_outputs)
    train_mae = mean_absolute_error(train_targets, train_outputs)

    model.eval()
    test_loss = 0
    test_outputs = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            loss = criterion(out[batch.test_mask], batch.y[batch.test_mask])
            test_loss += loss.item()
            test_outputs.append(out[batch.test_mask].detach().cpu().numpy())
            test_targets.append(batch.y[batch.test_mask].detach().cpu().numpy())
    
    avg_test_loss = test_loss / len(test_loader)
    test_outputs = np.concatenate(test_outputs, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)
    test_r2 = r2_score(test_targets, test_outputs)
    test_mae = mean_absolute_error(test_targets, test_outputs)
    
    print(f'Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Train R^2: {train_r2:.4f}, Train MAE: {train_mae:.4f}, Test Loss: {avg_test_loss:.4f}, Test R^2: {test_r2:.4f}, Test MAE: {test_mae:.4f}')
    if epoch>100:
        scheduler.step(avg_test_loss)


torch.save(model.state_dict(), 'gat_model.pth')