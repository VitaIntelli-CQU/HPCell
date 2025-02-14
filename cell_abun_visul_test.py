import os
import torch
import warnings
warnings.filterwarnings('ignore')

gpu_list = [1]
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

case = "A50"

# Load and initialize model
model = HPCell(vit_depth=3)
checkpoint_path = f"/home/zhaoyk/Hist2Cell/Hist2Cell/model_weights/{case}.pth"
model.load_state_dict(torch.load(checkpoint_path))
model = model.to(device)

test_slides_path = f"/home/zhaoyk/Hist2Cell/Hist2Cell/train_test_splits/humanlung_cell2location/test_leave_{case}.txt"
test_slides = [line.strip() for line in open(test_slides_path).readlines()]

# Import required libraries
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Batch
from tqdm import tqdm
import numpy as np
from scipy.stats import pearsonr
from PIL import Image
import matplotlib.pyplot as plt
import joblib
import os
import torch_geometric

torch_geometric.typing.WITH_PYG_LIB = False

# Load test dataset
def load_test_dataset(slides, base_path):
    graph_list = [torch.load(os.path.join(base_path, f"{item}.pt")) for item in slides]
    return Batch.from_data_list(graph_list)

base_path = "/home/zhaoyk/Hist2Cell/Hist2Cell/example_data/humanlung_cell2location"
test_dataset = load_test_dataset(test_slides, base_path)

# Configure DataLoader
hop = 2
subgraph_bs = 16
test_loader = NeighborLoader(
    test_dataset,
    num_neighbors=[-1] * hop,
    batch_size=subgraph_bs,
    directed=False,
    input_nodes=None,
    shuffle=False,
    num_workers=2,
)

# Evaluate model
with torch.no_grad():
    model.eval()
    test_cell_pred_list, test_cell_label_list, test_cell_pos_list = [], [], []

    for graph in tqdm(test_loader, desc="Evaluating"):
        x, y, pos = graph.x.to(device), graph.y.to(device), graph.pos.to(device)
        edge_index = graph.edge_index.to(device)

        cell_label = y[:, 250:]
        cell_pred = model(x, edge_index, pos)

        center_num = len(graph.input_id)
        test_cell_label_list.append(cell_label[:center_num].cpu().numpy())
        test_cell_pred_list.append(cell_pred[:center_num].cpu().numpy())
        test_cell_pos_list.append(pos[:center_num].cpu().numpy())

    # Concatenate results
    test_cell_pred_array = np.concatenate(test_cell_pred_list, axis=0)
    test_cell_label_array = np.concatenate(test_cell_label_list, axis=0)
    test_cell_pos_array = np.concatenate(test_cell_pos_list, axis=0)

# Compute Pearson Correlation and organize predictions
Predictions = {}
for slide_no, slide_name in enumerate(test_slides):
    indices = np.where(test_dataset.batch.cpu().numpy() == slide_no)[0]

    preds_sub = test_cell_pred_array[indices]
    labels_sub = test_cell_label_array[indices]
    pos_sub = test_cell_pos_array[indices]

    pearson_avg = np.mean([pearsonr(preds_sub[:, i], labels_sub[:, i])[0] for i in range(preds_sub.shape[1])])

    Predictions[slide_name] = {
        'cell_abundance_predictions': preds_sub,
        'cell_abundance_labels': labels_sub,
        'coords': pos_sub,
    }

    print(f"Slide {slide_name} has PCC: {pearson_avg:.6f}")

# Visualization
visualize_cell_names = ['Ciliated', 'Basal', 'AT1', 'AT2', 'Chondrocyte', 'T_reg', 'Suprabasal', 'SMG_Mucous',
                        'B_naive', 'Fibro_adventitial', 'Fibro_peribronchial']
cell_types = joblib.load("/home/zhaoyk/Hist2Cell/Hist2Cell/example_data/humanlung_cell2location/cell_types.pkl")

slide = "WSA_LngSP9258467"
hist2cell_abundances = np.clip(Predictions[slide]['cell_abundance_predictions'], a_min=0, a_max=None)
coordinates = Predictions[slide]['coords']
X, Y = coordinates[:, 0] / 4, coordinates[:, 1] / 4

image_path = "/home/zhaoyk/Hist2Cell/Hist2Cell/example_data/example_raw_data/WSA_LngSP9258467/WSA_LngSP9258467_low_res.jpg"
img = Image.open(image_path)

# Define function for plotting
def plot_cell_abundance(img, X, Y, cell_abundance, cell_name, save_dir, cmap='viridis', s=95):
    fig, ax = plt.subplots()
    ax.imshow(img.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT))
    sc = ax.scatter(img.size[0] - X, img.size[1] - Y, c=cell_abundance, s=s, alpha=0.8, cmap=cmap)
    ax.set_xlim(img.size[0] - (min(X) - 50), img.size[0] - (max(X) + 50))
    ax.set_ylim(img.size[1] - (min(Y) - 50), img.size[1] - (max(Y) + 50))
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xticks([])
    ax.set_yticks([])

    cbar = plt.colorbar(sc, ax=ax)
    cbar.ax.tick_params(labelsize=14)
    ax.set_title(cell_name, fontsize=16)

    save_path = os.path.join(save_dir, f"{cell_name}.png")
    plt.savefig(save_path)
    plt.show()

# Generate plots for each cell type
output_dir = "/home/zhaoyk/Hist2Cell/Hist2Cell/cell_abun_visual_A50/"
os.makedirs(output_dir, exist_ok=True)

for cell_name in tqdm(visualize_cell_names, desc="Visualizing"):
    cell_idx = cell_types.index(cell_name)
    plot_cell_abundance(img, X, Y, hist2cell_abundances[:, cell_idx], cell_name, output_dir)


