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
import sknetwork as skn
from torch.utils.data import Dataset

pad = tg.transforms.Pad(457,700)

class F_Dataset(tg.data.Dataset):
    def __init__(self, df, transform=None, pre_transform=None, use_bpps=False):
        super().__init__(df, transform, pre_transform)
        self.use_bpps = use_bpps
        df_a = df.loc[df.experiment_type == '2A3_MaP'].reset_index(drop=True)
        df_d = df.loc[df.experiment_type == 'DMS_MaP'].reset_index(drop=True)
        #self.bpps = pd.read_csv('train_files/bpps.csv')
        # select train sequence
        self.seq = df_a['sequence'].values
        self.bpps_paths = df_a['paths'].values
        # self.paths = df_.values
        # select target values
        self.react_a = df_a[[c for c in df_a.columns if 'reactivity_0' in c]].values
        self.react_d = df_d[[c for c in df_d.columns if 'reactivity_0' in c]].values
        # select target values error
        self.react_a_error = df_a[[c for c in df_a.columns if 'reactivity_error_0' in c]].values
        self.react_d_error = df_d[[c for c in df_d.columns if 'reactivity_error_0' in c]].values
        self.a_noise = df_a['signal_to_noise'].values
        self.d_noise = df_d['signal_to_noise'].values
        # self.gdc= tg.transforms.GDC(self_loop_weight=1,
        #                                    normalization_in='sym',
        #                                    normalization_out='col',
        #                                    diffusion_kwargs=dict(method='ppr', alpha=0.05),
        #                                    sparsification_kwargs=dict(method='topk', k=3, dim=0),
        #                                    exact=True, )

        del df_a, df_d

        gc.collect()

    def len(self):
        return len(self.seq)

    def loadbpps(self, bpp_path):
        content = pd.read_csv(bpp_path, header=None, delimiter=' ')
        return content

    def get(self, item):
        # prepare sequence
        seq, bpp_path = self.seq[item], self.bpps_paths[item]
        # seq = self.seq[item]
        ss, mfe = RNA.fold(seq)
        node_features = graph_utils.ohe_seq(seq)
        g = graph_utils.dotbracket_to_graph(ss)
        edges = list(g.edges(data=True))
        edge_attr = torch.Tensor([[0, 1] if e[2]['edge_type'] == 'adjacent' else [1, 0] for e in
                                   edges])
        edge_index = list(g.edges())

        # stack target reactivities
        target = torch.from_numpy(np.stack([self.react_a[item], self.react_d[item]], -1))
        # stack target error
        target_error = torch.from_numpy(np.stack([self.react_a_error[item], self.react_d_error[item]], -1))
        # noise
        noise = torch.FloatTensor([self.a_noise[item], self.d_noise[item]])

        if self.use_bpps == True:
            bpps = self.loadbpps(bpp_path)

            edge_index =pd.DataFrame(edge_index)
            bpps = pd.merge(edge_index, bpps, how='left').fillna(0).loc[:, 2].values
            edge_index = torch.LongTensor(edge_index.values).t().contiguous()
            edge_attr = torch.concatenate([edge_attr,torch.FloatTensor(bpps).unsqueeze(dim=-1)],dim=1)
            data = tg.data.Data(x=node_features,
                                edge_index=edge_index,
                                edge_attr = edge_attr,
                                y=target,
                                target_error=target_error,
                                noise=noise)
            del g

            return data

        # edge_index = list(g.edges())
        edge_index = pd.DataFrame(edge_index)
        edge_index = torch.LongTensor(edge_index.values).t().contiguous()


        del g

        data = tg.data.Data(x=node_features,
                            edge_index=edge_index,
                            # bpps=bpps,
                            y=target,
                            target_error=target_error,
                            noise=noise)

        return data


class Bpps_Dataset(tg.data.Dataset):
    def __init__(self, df, transform=None, pre_transform=None):
        super().__init__(df, transform, pre_transform)
        df_a = df.loc[df.experiment_type == '2A3_MaP'].reset_index(drop=True)
        df_d = df.loc[df.experiment_type == 'DMS_MaP'].reset_index(drop=True)
        self.seq = df_a['sequence'].values
        self.bpps_paths = df_a['paths'].values
        self.structures = df_a['structure'].values
        # select target values
        self.react_a = df_a[[c for c in df_a.columns if 'reactivity_0' in c]].values
        self.react_d = df_d[[c for c in df_d.columns if 'reactivity_0' in c]].values
        # select target values error
        self.lh = pd.read_parquet('train_files/lh.parquet')
        self.lh.rename(columns={'n_1': '0_x', 'n_2': '0_y'}, inplace=True)
        self.lh.drop('count', axis=1, inplace=True)

        del df_a, df_d
        gc.collect()

    def len(self):
        return len(self.seq)

    def loadbpps(self, bpp_path):
        content = pd.read_csv(bpp_path, header=None, delimiter=' ')
        content[[0, 1]] = content[[0, 1]] - 1
        return content

    def get(self, item):
        # loading seq bpp
        seq, bpp_path = self.seq[item], self.bpps_paths[item]
        ss = self.structures[item]

        # seq df
        seq_df = pd.DataFrame([x for x in seq]).reset_index()

        # constructing graph
        node_features = pd.DataFrame(graph_utils.ohe_seq(seq))
        g = construct_graph(sequence=seq, dotbracket=ss, edge_construction_funcs=edge_funcs_1)
        edge_attr = torch.Tensor(graph_utils.edge_attributes(g))

        # stack target reactivities
        target = torch.from_numpy(np.stack([self.react_a[item], self.react_d[item]], -1))

        bpps = self.loadbpps(bpp_path)

        # merge edge features with bpps
        edge_features = pd.DataFrame(list(g.edges()))
        edge_features = pd.merge(edge_features, bpps, how='left', left_on=[0, 1], right_on=[0, 1]).fillna(0)
        # merge edge features with LH
        edge_features.rename(columns={0: 'e_1', 1: 'e_2'}, inplace=True)
        edge_features = edge_features.merge(seq_df, left_on='e_1', right_on=['index'], how='left').drop('index',
                                                                                                        axis=1).merge(
            seq_df, how='left', left_on='e_2', right_on=['index']).drop('index', axis=1)

        edge_features = edge_features.merge(self.lh, how='left', left_on=['e_1', 'e_2', '0_x', '0_y'],
                                            right_on=['e_1', 'e_2', '0_x', '0_y']
                                            )
        # final edge attr BPPS + LH + P
        edge_attr = torch.concatenate(
            [edge_attr, torch.from_numpy(edge_features[[2, 'p', 'lh']].values.astype(np.float32))], dim=1)

        # edge_index,edge_attr = tg.utils.to_undirected(edge_index, edge_attr)
        data = tg.data.Data(x=torch.from_numpy(node_features.values.astype(np.float32)),
                            edge_index=torch.LongTensor(list(g.edges())).t().contiguous(),
                            edge_attr=edge_attr,
                            y=target,
                            )

        return data


class F_Test_Dataset(tg.data.Dataset):
    def __init__(self,df,transform = None,pre_transform = None,use_bpps=False):
        super().__init__(df,transform,pre_transform)
        self.df = df
        self.use_bpps = use_bpps
        self.bpps_paths = self.df['paths'].values
        # self.gdc = tg.transforms.GDC(self_loop_weight=1,
        #                                    normalization_in='sym',
        #                                    normalization_out='col',
        #                                    diffusion_kwargs=dict(method='ppr', alpha=0.05),
        #                                    sparsification_kwargs=dict(method='topk', k=3, dim=0),
        #                                    exact=True, )

    def len(self):
        return len(self.df)

    def loadbpps(self, bpp_path):
        content = pd.read_csv(bpp_path, header=None, delimiter=' ')
        return content


    def get(self,item):
        # prepare sequence
        id_min,id_max,seq = self.df.loc[item,['id_min','id_max','sequence']]
        bpp_path = self.bpps_paths[item]
        ss, mfe = RNA.fold(seq)
        node_features = graph_utils.ohe_seq(seq)
        g = graph_utils.dotbracket_to_graph(ss)
        edges = list(g.edges(data=True))
        edge_attr = torch.Tensor([[0, 1] if e[2]['edge_type'] == 'adjacent' else [1, 0] for e in
                                 edges])
        edge_index =list(g.edges())

        if self.use_bpps == True:

            bpps = self.loadbpps(bpp_path)

            # edge_index = torch.LongTensor(edge_index.values).t().contiguous()
            edge_index = pd.DataFrame(edge_index)
            bpps = pd.merge(edge_index, bpps, how='left').fillna(0).loc[:, 2].values
            edge_index = torch.LongTensor(edge_index.values).t().contiguous()
            edge_attr = torch.concatenate([edge_attr, torch.FloatTensor(bpps).unsqueeze(dim=-1)], dim=1)
            data = tg.data.Data(x=node_features,
                                edge_index=edge_index,
                                edge_attr=edge_attr
                                )
            del g

            return data

        edge_index = torch.LongTensor(edge_index).t().contiguous()


        data = tg.data.Data(x=node_features,
                    edge_index=edge_index,
                    #edge_attr=edge_attr,
                    )

        return data


class Test_Bpps_Dataset(tg.data.Dataset):
    def __init__(self,df,transform = None,pre_transform = None):
        super().__init__(df,transform,pre_transform)
        self.df = df
        self.bpps_paths = self.df['paths'].values
        self.structures = self.df['structure'].values
        self.lh = pd.read_parquet('train_files/lh.parquet')
        self.lh.rename(columns={'n_1': '0_x', 'n_2': '0_y'}, inplace=True)
        self.lh.drop('count', axis=1, inplace=True)

    def len(self):
        return len(self.df)

    def loadbpps(self, bpp_path):
        content = pd.read_csv(bpp_path, header=None, delimiter=' ')
        content[[0, 1]] = content[[0, 1]] - 1
        return content


    def get(self,item):
        # prepare sequence
        id_min,id_max,seq = self.df.loc[item,['id_min','id_max','sequence']]
        bpp_path = self.bpps_paths[item]

        ss = self.structures[item]
        node_features = pd.DataFrame(graph_utils.ohe_seq(seq))
        g = construct_graph(sequence=seq, dotbracket=ss, edge_construction_funcs=edge_funcs_1)

        edge_attr = torch.Tensor(graph_utils.edge_attributes(g))
        seq_df = pd.DataFrame([x for x in seq]).reset_index()


        bpps = self.loadbpps(bpp_path)



        edge_features = pd.DataFrame(list(g.edges()))
        edge_features = pd.merge(edge_features, bpps, how='left', left_on=[0, 1], right_on=[0, 1]).fillna(0)

        edge_features.rename(columns={0: 'e_1', 1: 'e_2'}, inplace=True)

        edge_features = edge_features.merge(seq_df, left_on='e_1', right_on=['index'], how='left').drop('index',axis=1).merge(
            seq_df, how='left', left_on='e_2', right_on=['index']).drop('index', axis=1)

        edge_features = edge_features.merge(self.lh, how='left', left_on=['e_1', 'e_2', '0_x', '0_y'],
                                            right_on=['e_1', 'e_2', '0_x', '0_y']
                                            )
        edge_attr = torch.concatenate(
            [edge_attr, torch.from_numpy(edge_features[[2, 'p', 'lh']].values.astype(np.float32))], dim=1)

        # edge_index, edge_attr = tg.utils.to_undirected(edge_index, edge_attr)
        data = tg.data.Data(x=torch.from_numpy(node_features.values.astype(np.float32)),
                            edge_index=torch.LongTensor(list(g.edges())).t().contiguous(),
                            edge_attr=edge_attr,
                            )


        return data


class Graph_Dataset(tg.data.Dataset):
    def __init__(self, df, transform=None, pre_transform=None):
        super().__init__(df, transform, pre_transform)
        df_a = df.loc[df.experiment_type == '2A3_MaP'].reset_index(drop=True)
        df_d = df.loc[df.experiment_type == 'DMS_MaP'].reset_index(drop=True)
        self.seq = df_a['sequence'].values
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
        ss = self.structures[item]
        #emb = torch.LongTensor([self.nucleotid_mapper[x] for x in seq])
        # constructing graph
        node_features = graph_utils.ohe_seq(seq)
        g = construct_graph(sequence=seq, dotbracket=ss, edge_construction_funcs=edge_funcs_1)

        edge_attr = torch.from_numpy(np.load(f'train_files/clean_attributes/{item}.npy'))

        # stack target reactivities
        edge_index = torch.LongTensor(list(g.edges())).t().contiguous()
        edge_index, edge_attr = tg.utils.to_undirected(edge_index, edge_attr[:, :6])

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
        df_a = df.loc[df.experiment_type == '2A3_MaP'].reset_index(drop=True)
        df_d = df.loc[df.experiment_type == 'DMS_MaP'].reset_index(drop=True)
        self.seq = df_a['sequence'].values
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
        seq, bpp_path = self.seq[item], self.bpps_paths[item]
        ss = self.structures[item]


        node_features = graph_utils.ohe_seq(seq)
        g = construct_graph(sequence=seq, dotbracket=ss, edge_construction_funcs=edge_funcs_1)
        edge_attr = torch.from_numpy(np.load(f'train_files/valid_attributes/{item}.npy'))
        # stack target reactivities
        target = torch.from_numpy(np.stack([self.react_a[item], self.react_d[item]], -1))
        edge_index = torch.LongTensor(list(g.edges())).t().contiguous()
        edge_index, edge_attr = tg.utils.to_undirected(edge_index, edge_attr[:, :6])
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
        node_features = graph_utils.ohe_seq(seq)
        g = construct_graph(sequence=seq, dotbracket=ss, edge_construction_funcs=edge_funcs_1)

        edge_attr = torch.from_numpy(np.load(f'train_files/test_attributes/{item}.npy'))
        edge_index = torch.LongTensor(list(g.edges())).t().contiguous()
        edge_index, edge_attr = tg.utils.to_undirected(edge_index, edge_attr[:, :6])

        emb = torch.LongTensor([self.nucleotid_mapper[x] for x in seq])
        emb = torch.nn.functional.pad(emb, [0, 457 - len(seq)])
        mask = torch.zeros(457, dtype=torch.bool)
        data = tg.data.Data(x=node_features,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            emb = emb,
                            emb_mask = mask
                            )

        return data





# # #
# x_train = pd.read_parquet('train_files/test_seq_struct.parquet')
# ds = Test_Graph_Dataset(x_train)
#
# x = ds[0]
# print(x)
#



# #
# transform = tg.transforms.LocalDegreeProfile()
#
# #
#
#
# transform = tg.transforms.GDC()
# print(x)
# g = transform(x)
# print(g.edge_attr)
# # print(x.edge_attr)

