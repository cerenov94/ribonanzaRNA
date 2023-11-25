import torch_geometric.nn as gnn
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch_geometric.utils
import math




class GraphTransEncoderLayer(torch.nn.Module):
    def __init__(self,in_channels,dropout = 0.3,heads = None):
        super().__init__()
        self.in_channels = in_channels
        self.dropout_prob = dropout
        self.encoder = gnn.TransformerConv(in_channels=self.in_channels,
                                           out_channels=32,
                                           heads=self.in_channels//32,
                                           beta=True,
                                           edge_dim=self.in_channels,
                                           )
        self.lin1 = nn.Linear(self.in_channels,4*self.in_channels)
        self.lin2 = nn.Linear(4*self.in_channels,self.in_channels)
        self.norm1 = gnn.LayerNorm(self.in_channels)
        self.norm2 = gnn.LayerNorm(self.in_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self,x,edge_index,edge_attr):
        x = self.norm1(x + self.encoder(x,edge_index,edge_attr))
        x = self.norm2(x + self.ff(x))
        return x

    def ff(self,x):
        x = self.lin2(self.dropout(self.act(self.lin1(x))))
        return x



class AttentiveGraphNet(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            pe:int,
            hidden_channels: int,
            out_channels: int,
            edge_dim: int,
            num_layers: int,
            num_a_layers:int,
            dropout: float,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.pe = pe
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.dropout = dropout


        self.node_encoder = nn.Embedding(self.in_channels,self.hidden_channels//6*4)
        self.edge_encoder = nn.Linear(self.edge_dim,self.hidden_channels)
        self.pe_encoder = nn.Linear(self.pe,self.hidden_channels//6*2)
        self.pe_norm = nn.LayerNorm(self.pe)
        self.transformer = nn.ModuleList()
        for i in range(1,num_layers+1):
            layer = gnn.GPSConv(channels=self.hidden_channels,
                                conv=gnn.TransformerConv(in_channels=self.hidden_channels,
                                                         out_channels=32,
                                                         heads=self.hidden_channels//32,
                                                         beta=True,
                                                         edge_dim=self.hidden_channels
                                                         ),
                                act='gelu',
                                heads=self.hidden_channels//32,
                                dropout=self.dropout,
                                norm='layer',
                                #norm_kwargs={'in_channels':self.hidden_channels}
                                )
            self.transformer.append(layer)
        self.perfomer_layers = nn.ModuleList()
        for j in range(1,num_a_layers+1):
            layer = gnn.GPSConv(channels=self.hidden_channels,
                                conv=gnn.TransformerConv(in_channels=self.hidden_channels,
                                                         out_channels=32,
                                                         heads=self.hidden_channels // 32,
                                                         beta=True,
                                                         edge_dim=self.hidden_channels
                                                         ),
                                act='gelu',
                                heads=self.hidden_channels//32,
                                dropout=self.dropout,
                                norm='layer',
                                attn_type='performer',
                                )
            self.perfomer_layers.append(layer)
        self.norm = gnn.LayerNorm(self.hidden_channels)


        self.output = gnn.Linear(self.hidden_channels,self.out_channels)
    def forward(self,x,edge_index,edge_attr,batch,pe):
        x = self.node_encoder(x)
        pe = self.pe_encoder(self.pe_norm(pe))
        x = torch.concatenate([x,pe],dim=1)

        edge_attr = self.edge_encoder(edge_attr)
        for layer,perform_layer in zip(self.transformer,self.perfomer_layers):
            x = self.norm(layer(x,edge_index,batch=batch,edge_attr=edge_attr)+perform_layer(x,edge_index,batch=batch,edge_attr=edge_attr))
        return self.output(x)