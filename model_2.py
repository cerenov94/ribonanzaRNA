import torch_geometric.nn as gnn
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_geometric.utils
import math
from model_3 import GATEConv

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[...,None] * emb[None,...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AttentiveGraphNet(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            edge_dim: int,
            num_layers: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers

        self.node_encoder_2 = gnn.ChebConv(self.in_channels,self.hidden_channels,K=5)
        self.node_encoder_1 = gnn.Linear(self.in_channels,self.hidden_channels,bias=False)
        self.edge_encoder = nn.Linear(self.edge_dim,self.hidden_channels)
        self.pos_encoder = SinusoidalPosEmb(self.hidden_channels)

        self.trans_conv_1 = gnn.TransformerConv(self.hidden_channels,32,edge_dim=self.hidden_channels,heads=self.hidden_channels//32)
        self.lin_1 = nn.Linear(self.hidden_channels,self.hidden_channels)
        self.trans_conv_2 = gnn.TransformerConv(self.hidden_channels,32,edge_dim=self.hidden_channels,heads=self.hidden_channels//32)
        self.lin_2 = nn.Linear(self.hidden_channels,self.hidden_channels)
        self.norm1 = nn.LayerNorm(self.hidden_channels)
        self.norm2 = nn.LayerNorm(self.hidden_channels)
        self.norm3 = nn.LayerNorm(self.hidden_channels)
        self.norm4 = nn.LayerNorm(self.hidden_channels)
        self.act = nn.ReLU()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.hidden_channels,
                                       nhead=self.hidden_channels//32,
                                       dim_feedforward=4 * self.in_channels,
                                       dropout=0.5,
                                       activation='relu',
                                       batch_first=True,
                                       norm_first=True
                                       ),
            num_layers=self.num_layers
        )

        self.fc = nn.Linear(self.hidden_channels, self.out_channels)

    def forward(self,x,edge_index,edge_attr,batch):

        position = torch.arange(x.shape[0], device=x.device).unsqueeze(0)
        position = self.pos_encoder(position).squeeze(dim=0).squeeze(0)


        emb = self.node_encoder_1(x)
        n = self.node_encoder_2(x,edge_index)
        edge_attr = self.edge_encoder(edge_attr)

        # encoder 1
        x_1 = F.dropout(self.trans_conv_1(n,edge_index,edge_attr),p=0.1,training=self.training)
        n = self.norm1((n+x_1))
        x_1 = self.act(self.lin_1(n))
        n = F.dropout(self.norm2((n+x_1)),p=0.1,training=self.training)
        # encoder 2
        x_1 = F.dropout(self.trans_conv_2(n, edge_index, edge_attr), p=0.1, training=self.training)
        n = self.norm3((n + x_1))
        x_1 = self.act(self.lin_2(n))
        n = F.dropout(self.norm4((n + x_1)), p=0.1, training=self.training)

        n = emb+n+position
        n = self.encoder(n)
        n = self.fc(n)
        return n