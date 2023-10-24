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


class A_Module(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(A_Module,self).__init__()
        self.trans = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(in_channels, nhead=4, dim_feedforward=4 * in_channels), num_layers=4
        )
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.trans(x)
        x = self.linear(x)
        return x

class GNN(nn.Module):
    def __init__(self):
        super(GNN,self).__init__()
        self.hidden_channels =192

        self.node_encoder = gnn.ChebConv(4, self.hidden_channels,K=5)
        self.edge_encoder = nn.Linear(4,self.hidden_channels)


        self.layers = torch.nn.ModuleList()
        self.gru_layers = nn.ModuleList()
        self.res_layers = nn.ModuleList()
        self.a_module = A_Module(self.hidden_channels,self.hidden_channels)

        heads = 4
        for i in range(1,5):
            conv1 = gnn.NNConv(self.hidden_channels,self.hidden_channels,nn = MapE2NxN(self.hidden_channels,self.hidden_channels*self.hidden_channels,self.hidden_channels))
            # conv1 = gnn.GENConv(self.hidden_channels, self.hidden_channels, aggr='softmax',
            #                t=1.0, learn_t=True, num_layers=2, norm='layer',bias=True)
            #conv1 = GATEConv(self.hidden_channels,self.hidden_channels,dropout=0.1,edge_dim=self.hidden_channels)
            norm = nn.LayerNorm(self.hidden_channels, elementwise_affine=True)
            act = nn.ReLU(inplace=True)
            layer = gnn.DeepGCNLayer(conv1,norm,act,block='res+',dropout=0.1,ckpt_grad=i % 3)
            self.layers.append(layer)
        self.final_layer = gnn.GENConv(self.hidden_channels,2,learn_t=True,norm='layer',edge_dim=self.hidden_channels,bias=True)
        self.linear = nn.Linear(self.hidden_channels,2)
    def forward(self,x,edge_index,edge_attr,batch):
        x = self.node_encoder(x,edge_index)
        edge_attr = (self.edge_encoder(edge_attr))
        x = self.layers[0].conv(x, edge_index, edge_attr)
        for layer1 in self.layers[1:]:
            x = layer1(x,edge_index,edge_attr)
        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x,p=0.1,training=self.training)
        #x = F.relu_(self.final_layer(x,edge_index,edge_attr))
        x = self.linear(x)
        return x



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


