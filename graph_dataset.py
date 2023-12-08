import torch_geometric as tg
import pandas as pd

import torch
import numpy as np
import RNA
import gc
import graph_utils
import torch_geometric.transforms as T
from graphein.rna.graphs import construct_graph
from graphein.rna.edges import (
    add_pseudoknots,
    add_phosphodiester_bonds,
    add_base_pairing_interactions
)
from typing import List,Callable
edge_funcs_1: List[Callable] = [
    add_base_pairing_interactions,
    add_phosphodiester_bonds,
    add_pseudoknots,
]



class Graph_Dataset(tg.data.Dataset):
    def __init__(self, df, transform=None, pre_transform=None):
        super().__init__(df, transform, pre_transform)
        df['L'] = df['sequence'].apply(len)
        df = df[df['L'] >= 177].reset_index(drop=True)
        df_a = df.loc[df.experiment_type == '2A3_MaP'].reset_index(drop=True)
        df_d = df.loc[df.experiment_type == 'DMS_MaP'].reset_index(drop=True)
        m = (df_a['signal_to_noise'].values >= 0.8) & (df_d['signal_to_noise'].values >= 0.8)
        df_a = df_a.loc[m].reset_index(drop=True)
        df_d = df_d.loc[m].reset_index(drop=True)
        self.seq = df_a['sequence'].values
        self.seq_id = df_a['sequence_id'].values
        self.structures = df_a['structure'].values
        # select target values
        self.react_a = df_a[[c for c in df_a.columns if 'reactivity_0' in c]].values
        self.react_d = df_d[[c for c in df_d.columns if 'reactivity_0' in c]].values
        self.nucleotid_mapper = {'A':0,'G':1,'C':2,'U':3}
        del df_a, df_d
        gc.collect()

    def len(self):
        return len(self.seq)

    def get(self, item):
        # loading seq bpp
        seq= self.seq[item]
        seq_id = self.seq_id[item]
        ss = self.structures[item]
        node_features = torch.LongTensor([self.nucleotid_mapper[x] for x in seq])
        # constructing graph
        #node_features = graph_utils.ohe_seq(seq)
        g = construct_graph(sequence=seq, dotbracket=ss, edge_construction_funcs=edge_funcs_1)
        #
        edge_attr = torch.from_numpy(np.load(f'train_files/new_files/train_attr/{seq_id}.npy'))

        # stack target reactivities
        edge_index = torch.LongTensor(list(g.edges())).t().contiguous()
        edge_index, edge_attr = tg.utils.to_undirected(edge_index, edge_attr[:, :5])
        #edge_index = tg.utils.to_undirected(edge_index)

        target = torch.from_numpy(np.stack([self.react_a[item], self.react_d[item]], -1))

        #node_features,edge_index = pad(node_features,edge_index)

        data = tg.data.Data(x=node_features,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            y=target,
                            )

        return data
class Valid_Dataset(tg.data.Dataset):
    def __init__(self, df, transform=None, pre_transform=None):
        super().__init__(df, transform, pre_transform)
        df['L'] = df['sequence'].apply(len)
        df = df[df['L'] >= 177].reset_index(drop=True)
        df_a = df.loc[df.experiment_type == '2A3_MaP'].reset_index(drop=True)
        df_d = df.loc[df.experiment_type == 'DMS_MaP'].reset_index(drop=True)

        self.seq = df_a['sequence'].values
        self.seq_id = df_a['sequence_id'].values
        self.bpps_paths = df_a['paths'].values
        self.structures = df_a['structure'].values
        # select target values
        self.react_a = df_a[[c for c in df_a.columns if 'reactivity_0' in c]].values
        self.react_d = df_d[[c for c in df_d.columns if 'reactivity_0' in c]].values
        del df_a, df_d
        self.nucleotid_mapper = {'A':0,'G':1,'C':2,'U':3}
        gc.collect()

    def len(self):
        return len(self.seq)

    def get(self, item):
        # loading seq bpp
        seq = self.seq[item]
        seq_id = self.seq_id[item]
        ss = self.structures[item]

        node_features = torch.LongTensor([self.nucleotid_mapper[x] for x in seq])
        #node_features = graph_utils.ohe_seq(seq)
        g = construct_graph(sequence=seq, dotbracket=ss, edge_construction_funcs=edge_funcs_1)
        edge_attr = torch.from_numpy(np.load(f'train_files/new_files/valid_attr/{seq_id}.npy'))
        # stack target reactivities
        target = torch.from_numpy(np.stack([self.react_a[item], self.react_d[item]], -1))
        edge_index = torch.LongTensor(list(g.edges())).t().contiguous()
        edge_index, edge_attr = tg.utils.to_undirected(edge_index, edge_attr[:, :5])
        #edge_index = tg.utils.to_undirected(edge_index)
        data = tg.data.Data(x=node_features,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            y=target,
                            )

        return data


class Test_Graph_Dataset(tg.data.Dataset):
    def __init__(self, df, transform=None, pre_transform=None):
        super().__init__(df, transform, pre_transform)
        self.sequences = df['sequence'].values
        self.structures = df['structure'].values
        self.nucleotid_mapper = {'A':0,'G':1,'C':2,'U':3}

    def len(self):
        return len(self.sequences)

    def get(self, item):
        # prepare sequence
        seq = self.sequences[item]

        ss = self.structures[item]
        node_features = torch.LongTensor([self.nucleotid_mapper[x] for x in seq])
        #node_features = graph_utils.ohe_seq(seq)
        g = construct_graph(sequence=seq, dotbracket=ss, edge_construction_funcs=edge_funcs_1)
        #
        edge_attr = torch.from_numpy(np.load(f'train_files/test_attributes/{item}.npy'))
        edge_index = torch.LongTensor(list(g.edges())).t().contiguous()
        edge_index, edge_attr = tg.utils.to_undirected(edge_index, edge_attr[:, :5])
        #edge_index = tg.utils.to_undirected(edge_index)

        # emb = torch.LongTensor([self.nucleotid_mapper[x] for x in seq])
        # emb = torch.nn.functional.pad(emb, [0, 457 - len(seq)])
        # mask = torch.zeros(457, dtype=torch.bool)
        data = tg.data.Data(x=node_features,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            # emb = emb,
                            # emb_mask = mask
                            )

        return data



