import random
import torch
import os
import numpy as np
import torch.utils.data



def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

setup_seed(3407)

import os
import torch
import warnings
warnings.filterwarnings('ignore')

gpu_list = [2]
gpu_list_str = ','.join(map(str, gpu_list))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from torch.nn import Linear
import torch.nn as nn
import torchvision.models as models
from torch_geometric.nn import GATv2Conv, LayerNorm
import sys, os

sys.path.append(os.path.dirname(os.getcwd()))
from model.ViT import Mlp, VisionTransformer
from torch_geometric.nn import HypergraphConv
from torch_geometric.data import Data
import torch.nn.functional as F
import timm

class HypergraphNeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(HypergraphNeuralNetwork, self).__init__()
        self.conv1 = HypergraphConv(input_dim, hidden_dim1)
        self.conv2 = HypergraphConv(hidden_dim1, output_dim)
        #self.conv3 = HypergraphConv(hidden_dim2, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index, data.edge_attr)
        x1 = F.dropout(F.relu(x), 0.3, training=True)
        x2 = self.conv2(x1, data.edge_index, data.edge_attr)
        #x2 = F.relu(x2)
        #x3 = self.conv3(x2, data.edge_index, data.edge_attr)
        x4 = self.norm(x2)

        return x4


def compute_adjacency_matrix(pos, adjacency_matrix):
    num_edges = adjacency_matrix.size(1)
    adjacency_matrix2 = torch.zeros((pos.size(0), pos.size(0)), dtype=torch.float)

    for edge_index in range(num_edges):
        node_a, node_b = adjacency_matrix[:, edge_index]
        distance = torch.norm(pos[node_a] - pos[node_b])
        adjacency_matrix2[node_a, node_b] = distance
        adjacency_matrix2[node_b, node_a] = distance

    max_distance = torch.max(adjacency_matrix2)
    adjacency_matrix2 = adjacency_matrix2 / max_distance

    nonzero_indices = torch.nonzero(adjacency_matrix2.sum(dim=1) > 0).squeeze()
    adjacency_matrix2 = adjacency_matrix2[nonzero_indices, :]
    adjacency_matrix2 = adjacency_matrix2[:, nonzero_indices]

    return adjacency_matrix2


def build_adj_hypergraph(features, adjacency_matrix, pos, threshold=0):
    n, m = features.size()
    device=features.device

    hypergraph_edges = []
    edge_weights = []

    adjacency_matrix2 = compute_adjacency_matrix(pos, adjacency_matrix)

    e, f = adjacency_matrix2.size()

    for i in range(e):
        Neighbor_distances = adjacency_matrix2[i, :]

        valid_neighbors = (Neighbor_distances > threshold).nonzero(as_tuple=True)[0]

        for neighbor in valid_neighbors:
            hypergraph_edges.append([i, neighbor.item()])
            edge_weights.append(Neighbor_distances[neighbor].item())

    hypergraph_edges = torch.tensor(hypergraph_edges, dtype=torch.long, device=device).t()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float, device=device)

    data = Data(x=features, edge_index=hypergraph_edges, edge_attr=edge_weights, y=None)

    return data


tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=False, checkpoint_path="/home/zhaoyk/Hist2Cell/Hist2Cell/provgigapath/pytorch_model.bin")

class HPCell(nn.Module):
    def __init__(self, cell_dim=80, vit_depth=3):
        super().__init__()
        
        self.tile_encoder = tile_encoder  # Feature extraction encoder
        
        # Define dimensions and configuration
        self.dimension = 32 * 8
        self.num_heads = 8
        self.dropout_rate = 0.3

        # Initialize Hypergraph Neural Network layer
        self.hypergraph_layer = HypergraphNeuralNetwork(512, 128, 64, 256)

        # Transformer for cell-level processing
        self.transformer_model = VisionTransformer(
            num_classes=cell_dim,
            embed_dim=self.dimension,
            depth=vit_depth,
            mlp_head=True,
            drop_rate=self.dropout_rate,
            attn_drop_rate=self.dropout_rate
        )

        # Fully connected layers for spot features
        self.spot_fc_initial = Linear(1536, 512)
        self.spot_fc_final = Linear(512, 256)

        # Prediction heads
        self.prediction_heads = nn.ModuleDict({
            'spot': Mlp(256, 1024, cell_dim),
            'local': Mlp(256, 1024, cell_dim),
            'fusion': Mlp(256, 1024, cell_dim)
        })

    def forward(self, input_data, edges, positions):
        # Extract spot features using encoder (frozen during inference)
        with torch.no_grad():
            self.tile_encoder.eval()
            spot_features = self.tile_encoder(input_data).squeeze()

        # Process spot features through FC layers
        spot_features_transformed = self.spot_fc_initial(spot_features)

        # Create hypergraph structure and extract local features
        hypergraph_input = build_adj_hypergraph(spot_features_transformed, edges, positions)
        local_features = self.hypergraph_layer(hypergraph_input).unsqueeze(0)

        # Refine spot features
        refined_spot_features = self.spot_fc_final(spot_features_transformed)

        # Generate predictions from individual components
        spot_prediction = self.prediction_heads['spot'](refined_spot_features)
        local_features_squeezed = local_features.squeeze(0)
        local_prediction = self.prediction_heads['local'](local_features_squeezed)

        # Use transformer for global prediction
        global_prediction, global_features = self.transformer_model(local_features)
        global_prediction = global_prediction.squeeze()
        global_features = global_features.squeeze()

        # Combine features and compute fused prediction
        combined_features = torch.mean(torch.stack([refined_spot_features, local_features_squeezed, global_features]), dim=0)
        fused_prediction = self.prediction_heads['fusion'](combined_features)

        # Average all predictions for final output
        final_cell_prediction = torch.mean(torch.stack([
            spot_prediction, local_prediction, global_prediction, fused_prediction
        ]), dim=0)

        return final_cell_prediction


model = HPCell(vit_depth=3).to(device)

# Load train and test slide files
train_slides = [line.strip() for line in open("/home/zhaoyk/Hist2Cell/Hist2Cell/train_test_splits/humanlung_cell2location/train_leave_A50.txt")]
test_slides = [line.strip() for line in open("/home/zhaoyk/Hist2Cell/Hist2Cell/train_test_splits/humanlung_cell2location/test_leave_A50.txt")]

from torch_geometric.data import Batch

def load_graphs(slides, data_path):
    graph_list = [torch.load(os.path.join(data_path, f"{item}.pt")) for item in slides]
    return Batch.from_data_list(graph_list)

base_path = "/home/zhaoyk/Hist2Cell/Hist2Cell/example_data/humanlung_cell2location"
train_dataset = load_graphs(train_slides, base_path)
test_dataset = load_graphs(test_slides, base_path)

from torch_geometric.loader import NeighborLoader
import torch_geometric
torch_geometric.typing.WITH_PYG_LIB = False

hop = 2
subgraph_bs = 16
loader_args = {
    "num_neighbors": [-1] * hop,
    "batch_size": subgraph_bs,
    "directed": False,
    "num_workers": 2,
}

train_loader = NeighborLoader(train_dataset, shuffle=True, input_nodes=None, **loader_args)
test_loader = NeighborLoader(test_dataset, shuffle=False, input_nodes=None, **loader_args)

import numpy as np
from scipy.stats import pearsonr
import time

lr = 1e-4
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

num_epochs = 5
best_pearson = 0.0
start_time = time.time()

for epoch in range(num_epochs):
    print(f"{'-' * 50}\nEpoch: {epoch + 1}")
    print(f"Learning Rate: {optimizer.param_groups[0]['lr']}")

    model.train()
    train_metrics = {"loss": 0.0, "samples": 0, "preds": [], "labels": []}

    for batch in train_loader:
        x, y, edge_index, pos = batch.x.to(device), batch.y.to(device), batch.edge_index.to(device), batch.pos.to(device)
        cell_labels = y[:, 250:]

        optimizer.zero_grad()
        cell_preds = model(x, edge_index, pos)
        loss = criterion(cell_preds, cell_labels)
        loss.backward()
        optimizer.step()

        center_size = len(batch.input_id)
        train_metrics["loss"] += loss.item() * center_size
        train_metrics["samples"] += center_size
        train_metrics["preds"].append(cell_preds[:center_size].cpu().detach().numpy())
        train_metrics["labels"].append(cell_labels[:center_size].cpu().detach().numpy())

    train_loss = train_metrics["loss"] / train_metrics["samples"]
    train_preds = np.concatenate(train_metrics["preds"], axis=0)
    train_labels = np.concatenate(train_metrics["labels"], axis=0)

    train_pearson_avg = np.mean([pearsonr(train_preds[:, i], train_labels[:, i])[0] for i in range(train_preds.shape[1])])
    scheduler.step()

    model.eval()
    test_metrics = {"loss": 0.0, "samples": 0, "preds": [], "labels": []}

    with torch.no_grad():
        for batch in test_loader:
            x, y, edge_index, pos = batch.x.to(device), batch.y.to(device), batch.edge_index.to(device), batch.pos.to(device)
            cell_labels = y[:, 250:]
            cell_preds = model(x, edge_index, pos)

            loss = criterion(cell_preds, cell_labels)
            center_size = len(batch.input_id)

            test_metrics["loss"] += loss.item() * center_size
            test_metrics["samples"] += center_size
            test_metrics["preds"].append(cell_preds[:center_size].cpu().detach().numpy())
            test_metrics["labels"].append(cell_labels[:center_size].cpu().detach().numpy())

    test_loss = test_metrics["loss"] / test_metrics["samples"]
    test_preds = np.concatenate(test_metrics["preds"], axis=0)
    test_labels = np.concatenate(test_metrics["labels"], axis=0)

    test_pearson_avg = np.mean([pearsonr(test_preds[:, i], test_labels[:, i])[0] for i in range(test_preds.shape[1])])

    if test_pearson_avg > best_pearson:
        best_pearson = test_pearson_avg
        torch.save(model.state_dict(), "/home/zhaoyk/Hist2Cell/Hist2Cell/model_weights/A50.pth")
        print(f"Model saved with best test Pearson correlation: {test_pearson_avg:.6f}")

    elapsed_time = time.time() - start_time
    print(f"Training Time: {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")
    print(f"Train Loss: {train_loss:.6f}, Train Pearson Avg: {train_pearson_avg:.6f}")
    print(f"Test Loss: {test_loss:.6f}, Test Pearson Avg: {test_pearson_avg:.6f}")

