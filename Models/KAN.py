from efficient_kan import KANLinear
import torch
import torch.nn as nn
import torch.optim
import warnings

warnings.filterwarnings('ignore')
torch.cuda.empty_cache()


class KanMolde(nn.Module):
    def __init__(self):
        super(KanMolde, self).__init__()
        # self.fc1 = KANLinear(142, 128)
        self.fc1 = KANLinear(140, 128)
        self.fc2 = KANLinear(128, 128)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(128)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.norm(out)
        out = self.fc2(out)
        out = self.relu(out) 
        
        # out = self.sig(out)

        return out

