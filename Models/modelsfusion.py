import torch
from torch import nn
from efficient_kan import KANLinear

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

feature_dim = 2


class CModel(nn.Module):
    def __init__(self):
        super(CModel, self).__init__()

        self.transform_fc_1 = None

        # Add a weight for each model's output
        self.weight_transform = nn.Parameter(torch.ones(1))
        self.weight_gat = nn.Parameter(torch.ones(1))
        self.weight_kan = nn.Parameter(torch.ones(1))
        self.weight_se3 = nn.Parameter(torch.ones(1))

        # Gating mechanism
        self.gate_transform = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.gate_gat = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.gate_kan = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.gate_se3 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))


        self.fc1 = KANLinear(128+128+128+128, 384)
        self.fc2 = KANLinear(384, 128)
        self.fc3 = KANLinear(128, 8)
        self.fc4 = KANLinear(8+feature_dim, 1)  
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(384)
        self.sig = nn.Sigmoid()

    def forward(self, y_transform, y_gat, y_kan, y_se3, features):
        device = next(self.parameters()).device 

        out_transform = y_transform.to(device)
        out_gat = y_gat.to(device)
        out_kan = y_kan.to(device)
        out_se3 = y_se3.to(device)
        features = features.to(device)

        # Dynamically build self.transform_fc_1
        if self.transform_fc_1 is None or self.transform_fc_1.in_features != out_transform.shape[1]:
            self.transform_fc_1 = nn.Linear(out_transform.shape[1], 128).to(device)
    
        out_transform = self.transform_fc_1(out_transform)
        out_transform = self.relu(out_transform)
    
        gate_transform = self.gate_transform(out_transform)
        gate_gat = self.gate_gat(out_gat)
        gate_kan = self.gate_kan(out_kan)
        gate_se3 = self.gate_se3(out_se3)
        gates = torch.softmax(torch.stack([gate_transform, gate_gat, gate_kan, gate_se3], dim=1), dim=1)
        
        weights = torch.softmax(torch.stack((self.weight_transform, self.weight_gat, self.weight_kan, self.weight_se3)),
                                dim=0)
        weight_transform, weight_gat, weight_kan, weight_se3 = weights[0], weights[1], weights[2], weights[3]

        out = torch.cat([weight_transform * gates[:, 0] * out_transform,
                       weight_gat * gates[:, 1] * out_gat,
                       weight_kan * gates[:, 2] * out_kan,
                       weight_se3 * gates[:, 3] * out_se3], dim=-1).to(device)

        out = self.fc1(out)
        out = self.relu(out)
        out = self.norm(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = torch.cat([out, features], dim=-1).to(device)
        out = self.fc4(out)

        out = self.sig(out)

        return out

