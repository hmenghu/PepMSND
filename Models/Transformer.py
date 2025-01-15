import torch
from torch import nn
import torch.nn.functional as F
import copy


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PositionalEncoding(nn.Module):
    def __init__(self, embed, pad_size, dropout):
        super(PositionalEncoding, self).__init__()
        # Create positional encoding tensor
        pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        pe[:, 0::2] = torch.sin(pe[:, 0::2])  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(pe[:, 1::2])   # Apply cosine to odd indices

        self.pe = nn.Parameter(pe, requires_grad=True)  # Convert the positional encoding tensor into a trainable parameter

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + self.pe[:x.size(1)] # Add positional encoding to input
        out = self.dropout(out)
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, scale=None, mask=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # Q*K^T
        if scale is not None:
            attention = attention * scale
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)

        self.attention = ScaledDotProductAttention()

        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)

        if mask is not None:
            mask = mask.repeat(self.num_head, 1, 1)
        scale = K.size(-1) ** -0.5
        context = self.attention(Q, K, V, scale, mask=mask)  # Scaled_Dot_Product_Attention
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()

        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x
        out = self.layer_norm(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class ConfigTrans(object):

    def __init__(self):
        self.model_name = 'Transformer'
        self.dropout = 0.5
        self.num_classes = 1  
        self.num_epochs = 100
        self.pad_size = 1
        self.learning_rate = 0.001
        """Modification point, mainly changing the string length"""
        self.embed = 1024
        self.dim_model = 1024

        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 8
        self.num_encoder = 2


config = ConfigTrans()


class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.postion_embedding = PositionalEncoding(config.embed, config.pad_size, config.dropout)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])
        self.before_list = []
        self.after_list = []



    def forward(self, x):
        out = self.postion_embedding(x)
        for encoder in self.encoders:
            out = encoder(out)

        out = out.view(out.size(0), -1)

        return out