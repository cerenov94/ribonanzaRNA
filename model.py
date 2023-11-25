import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
import torch_geometric.nn as gnn
from torch_geometric.utils import to_networkx
from graph_dataset import Graph_Dataset,Bpps_Dataset
import torch.nn.functional as F
import torch_geometric as tg
from torch_geometric.nn import GATConv, MessagePassing, global_add_pool
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import GRUCell, Linear, Parameter
from torch import Tensor
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax
from typing import Optional

class GATEConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.att_l = Parameter(torch.empty(1, out_channels))
        self.att_r = Parameter(torch.empty(1, in_channels))

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out + self.bias
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j @ self.att_l.t()).squeeze(-1)
        alpha_i = (x_i @ self.att_r.t()).squeeze(-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return self.lin2(x_j) * alpha.unsqueeze(-1)



class MapE2NxN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super(MapE2NxN, self).__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x




class GraphTransEncoderLayer(torch.nn.Module):
    def __init__(self,in_channels,dropout = 0.3,heads = None):
        super().__init__()
        self.in_channels = in_channels
        self.dropout_prob = dropout
        self.encoder = gnn.GATv2Conv(in_channels,32,heads=in_channels//32,negative_slope=0.1,edge_dim=in_channels)
        self.lin = nn.Linear(self.in_channels,self.in_channels)
        self.norm1 = nn.LayerNorm(self.in_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self,x,edge_index,edge_attr):
        x_1 = x
        x_1 = self.encoder(x_1,edge_index,edge_attr)
        x = self.dropout(self.act(self.norm1(x+x_1)))
        return x

class GNN(nn.Module):
    def __init__(self,in_channels=6,hidden_channels = 128,decoder_hidden=128,edge_dim=4,dropout=0.5,num_layers=8,num_attentive_layers=2):
        super(GNN,self).__init__()
        self.node_encoder = nn.Linear(in_channels,hidden_channels,bias=False)
        self.edge_encoder = nn.Linear(edge_dim,hidden_channels)
        self.decoder = torch.nn.ModuleList()
        self.encoder = torch.nn.ModuleList()
        for e in range(1,num_attentive_layers+1):
            self.encoder.append(GraphTransEncoderLayer(hidden_channels,dropout=dropout))

        self.out_encoder = nn.Linear(hidden_channels,decoder_hidden)
        for i in range(1,num_layers+1):
            conv = gnn.NNConv(decoder_hidden,decoder_hidden,nn = nn.Linear(hidden_channels,decoder_hidden*decoder_hidden))
            norm = nn.LayerNorm(decoder_hidden, elementwise_affine=True)
            act = nn.GELU()
            layer = gnn.DeepGCNLayer(conv,norm,act,block='res+',dropout=dropout)
            self.decoder.append(layer)
        self.output = nn.Linear(decoder_hidden,2)
    def forward(self,x,edge_index,edge_attr,batch):

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        for a_layer in self.encoder:
            x = a_layer(x,edge_index,edge_attr)
        #x = F.relu_(self.out_encoder(x))

        h = self.decoder[0].conv(x,edge_index,edge_attr)
        for layer1 in self.decoder:
            h = layer1((x+h),edge_index,edge_attr)
        x = self.decoder[0].act(self.decoder[0].norm(h))
        return self.output(x)



#
#
#
# # # # #
#
# x_train = pd.read_parquet('train_files/valid_with_structure.parquet')
# #trans = tg.transforms.VirtualNode()
# ds = Graph_Dataset(x_train)
# model = GAT(4,32,2,4)
# #
# inputs = ds[0]
# outputs = model(inputs.x,inputs.edge_index,inputs.edge_attr,inputs.x)
# print(outputs.shape)







#
# model = GNN()
# x_train = pd.read_parquet('valid.parquet')
#
# train_ds = Bpps_Dataset(x_train)
#
# # print(train_ds[0]['x'])
# # print(train_ds[0]['edge_index'])
# print(model(train_ds[0]))
#
#
# inputs = train_ds[0]
# #
# #
# outputs = model(inputs)
#
# print(outputs.shape)
#


