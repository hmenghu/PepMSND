import torch
from torch_geometric.nn import GATConv
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
import warnings


warnings.filterwarnings('ignore')
torch.cuda.empty_cache()


class GATModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=2):
        super(GATModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=2, concat=False)


    def forward(self, x, edge_index, batch):  
        out = F.relu(self.conv1(x, edge_index))
        out = F.dropout(out, training=self.training)
        out = self.conv2(out, edge_index)
        out = global_mean_pool(out, batch)
        out = F.relu(out)
        # out = F.sigmoid(out)

        return out
