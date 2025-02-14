import os
import torch
import warnings
warnings.filterwarnings('ignore')

gpu_list = [3]
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


from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
import torch_geometric

torch_geometric.typing.WITH_PYG_LIB = False
from torch_geometric.data import Batch
import numpy as np
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams

rcParams['pdf.fonttype'] = 42
import numpy as np
from scipy.spatial.distance import jensenshannon
from tkinter import font
import joblib

case = "A50"
hop = 2
subgraph_bs = 16
save_path = '/home/zhaoyk/Hist2Cell/Hist2Cell/key_cell_picture'
os.makedirs(save_path, exist_ok=True)

# Model and checkpoint loading
model = HPCell(vit_depth=3)
checkpoint = torch.load("/home/zhaoyk/Hist2Cell/Hist2Cell/model_weights/A50.pth")
model.load_state_dict(checkpoint)
model = model.to(device)

# Read test slides list
test_slides = open(f"/home/zhaoyk/Hist2Cell/Hist2Cell/train_test_splits/humanlung_cell2location/test_leave_{case}.txt").read().split('\n')

# Load the graphs
test_graph_list = [
    torch.load(os.path.join("/home/zhaoyk/Hist2Cell/Hist2Cell/example_data/humanlung_cell2location", f"{item}.pt"))
    for item in test_slides
]
test_dataset = Batch.from_data_list(test_graph_list)

# Initialize NeighborLoader for batching
test_loader = NeighborLoader(
    test_dataset,
    num_neighbors=[-1] * hop,
    batch_size=subgraph_bs,
    directed=False,
    input_nodes=None,
    shuffle=False,
    num_workers=2,
)

# Inference loop
model.eval()
test_cell_pred_array, test_cell_label_array, test_cell_pos_array = [], [], []

with torch.no_grad():
    for graph in tqdm(test_loader):
        x, y, pos, edge_index = graph.x.to(device), graph.y.to(device), graph.pos.to(device), graph.edge_index.to(device)
        cell_label = y[:, 250:]
        cell_pred = model(x, edge_index, pos)

        center_num = len(graph.input_id)
        center_cell_label = cell_label[:center_num, :]
        center_cell_pred = cell_pred[:center_num, :]
        center_cell_pos = pos[:center_num, :]

        test_cell_label_array.append(center_cell_label.squeeze().cpu().detach().numpy())
        test_cell_pred_array.append(center_cell_pred.squeeze().cpu().detach().numpy())
        test_cell_pos_array.append(center_cell_pos.squeeze().cpu().detach().numpy())

# Concatenate results
test_cell_pred_array = np.concatenate(test_cell_pred_array)
test_cell_label_array = np.concatenate(test_cell_label_array)
test_cell_pos_array = np.concatenate(test_cell_pos_array)

# Prepare for predictions
Predictions = {}
for slide_no in range(len(test_slides)):
    indices = np.where(test_dataset.batch.numpy() == slide_no)
    test_cell_pred_array_sub = test_cell_pred_array[indices]
    test_cell_label_array_sub = test_cell_label_array[indices]
    test_cell_pos_array_sub = test_cell_pos_array[indices]

    # Calculate Pearson correlation
    test_cell_abundance_all_pearson_average = np.mean([
        pearsonr(test_cell_pred_array_sub[:, i], test_cell_label_array_sub[:, i])[0]
        for i in range(test_cell_pred_array_sub.shape[1])
    ])

    Predictions[test_slides[slide_no]] = {
        'cell_abundance_predictions': test_cell_pred_array_sub,
        'cell_abundance_labels': test_cell_label_array_sub,
        'coords': test_cell_pos_array_sub,
    }

    print(f"slide {test_slides[slide_no]} has PCC: {test_cell_abundance_all_pearson_average:.6f}")

# Load cell types
cell_types = joblib.load("/home/zhaoyk/Hist2Cell/Hist2Cell/example_data/humanlung_cell2location/cell_types.pkl")

# Key cell types to visualize
key_cell_types = ['CD4_EM_Effector', 'Ciliated', 'gdT', 'CD8_EM']
slide = 'WSA_LngSP9258463'

# Get predictions and ground truth
hist2cell_abundances = np.clip(Predictions[slide]['cell_abundance_predictions'], a_min=0, a_max=None)
ground_truth_abundances = Predictions[slide]['cell_abundance_labels']

# Function to create plots for histograms
def plot_histogram(cell_count, infer_cell_count, xlab, ylab, title, compute_kl=True, equal=True, save_path=None):
    plt.rcParams.update({'font.size': 14})  # Increase font size
    cor = np.round(np.corrcoef(cell_count.flatten(), infer_cell_count.flatten()), 3)[0, 1]
    max_val = np.concatenate([cell_count.flatten(), infer_cell_count.flatten()]).max()
    title = title + f'\nPearson R: {cor}'

    if compute_kl:
        js = np.array([jensenshannon(cell_count[r, :], infer_cell_count[r, :]) for r in range(cell_count.shape[0])])
        js = np.mean(js[~np.isnan(js)])
        title = title + f'\nAverage JSD: {np.round(js, 2)}'

    x_bins = int(np.max(cell_count)) if np.max(cell_count) > 1 else 35
    fig, ax = plt.subplots()
    h = ax.hist2d(x=cell_count.flatten(), y=infer_cell_count.flatten(), bins=[x_bins, 35], norm=matplotlib.colors.LogNorm())

    ax.set_xlabel(xlab, fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)
    if equal:
        plt.gca().set_aspect('equal', adjustable='box')
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.title(title, fontsize=10)

    cbar = plt.colorbar(h[3], ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label('Frequency', fontsize=10)

    if save_path:
        plt.savefig(os.path.join(save_path, title.split('\n')[0] + '.png'), bbox_inches='tight')
    plt.close()

# Plot for each key cell type
for cell_type in tqdm(key_cell_types):
    plt.figure(figsize=(2, 2))
    cell_idx = cell_types.index(cell_type)
    x = np.expand_dims(hist2cell_abundances[:, [cell_idx]], axis=0)
    y = np.expand_dims(ground_truth_abundances[:, [cell_idx]], axis=0)
    rcParams['figure.figsize'] = 3, 3
    rcParams["axes.facecolor"] = "white"

    plot_histogram(
        cell_count=y, infer_cell_count=x, equal=True,
        title=cell_type, xlab='Ground-truth cell proportion', ylab='Inferred cell abundance', save_path=save_path
    )