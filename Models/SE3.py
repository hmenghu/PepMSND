import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter
from torch_cluster import radius_graph
from Bio.PDB import PDBParser  

class EquivariantLayer(nn.Module):  # SE-3 EquivariantLayer
    def __init__(self, hidden_dim, num_heads, cutoff_distance):
        super(EquivariantLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.cutoff_distance = cutoff_distance
        self.linear_q = nn.Linear(3, hidden_dim * num_heads) 
        self.linear_k = nn.Linear(3, hidden_dim * num_heads)
        self.linear_v = nn.Linear(3, hidden_dim * num_heads)
        self.linear_out = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.SiLU = nn.SiLU()
        self.dropout = nn.Dropout(p=0.5)

        # Initialize weights
        nn.init.xavier_uniform_(self.linear_q.weight)
        nn.init.xavier_uniform_(self.linear_k.weight)
        nn.init.xavier_uniform_(self.linear_v.weight)
        nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, positions, edge_index):
        row, col = edge_index

        # Calculate relative position
        relative_positions = positions[row] - positions[col]

        # Split into multiple heads
        q = self.linear_q(relative_positions).view(-1, self.num_heads, self.hidden_dim)
        k = self.linear_k(relative_positions).view(-1, self.num_heads, self.hidden_dim)
        v = self.linear_v(relative_positions).view(-1, self.num_heads, self.hidden_dim)

        # Attention scores
        attn_scores = (q * k).sum(dim=-1) / (self.hidden_dim ** 0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1)
        attn_scores = self.dropout(attn_scores)

        # Weighted sum of values
        weighted_values = attn_scores.unsqueeze(-1) * v
        weighted_values = weighted_values.view(-1, self.hidden_dim * self.num_heads)
        updated_positions = scatter(weighted_values, col, dim=0, dim_size=positions.size(0), reduce='mean')

        # Linear projection and activation
        updated_positions = self.linear_out(updated_positions)
        updated_positions = self.norm(updated_positions)
        updated_positions = self.SiLU(updated_positions)

        return updated_positions

class SE3Transformer(nn.Module):
    def __init__(self, hidden_dim, num_layers, num_heads, cutoff_distance):
        super(SE3Transformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.ev_layers = nn.ModuleList([
            EquivariantLayer(hidden_dim, num_heads, cutoff_distance) for _ in range(num_layers)
        ])
        self.embedding = nn.Linear(3, hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 128) # Set output dimensions
        )   
        self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        self.sig = nn.Sigmoid()

    def forward(self, positions, batch_indices):
        if batch_indices is None:
            batch_indices = torch.zeros(positions.size(0), dtype=torch.long, device=positions.device)

        features = self.embedding(positions)

        for i, layer in enumerate(self.ev_layers):

            edge_index = radius_graph(positions, r=layer.cutoff_distance, batch=batch_indices, loop=False)

            updated_positions = layer(positions, edge_index)
            features = self.layer_norm[i](features + updated_positions)


        molecule_representation = scatter(features, dim=0, index=batch_indices, reduce='mean')

        output = self.output(molecule_representation)
        
        # output = self.sig(output)

        return output

# Extract atomic coordinates
def extract_coordinates(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('X', pdb_file)
    coordinates = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coordinates.append(atom.get_coord())
    coordinates = np.array(coordinates, dtype=np.float32)
    return torch.tensor(coordinates)

if __name__ == '__main__':

    model = SE3Transformer(hidden_dim=128, num_layers=3, num_heads=4, cutoff_distance=5.0)

    coordinates = extract_coordinates("C:/Users/HUHAOMENG/Desktop/Peptide structure Dataset/344.pdb")
    print(coordinates.shape)
    batch_indices = None
    out = model(coordinates, batch_indices)
    print(out.shape)
    print(out)
