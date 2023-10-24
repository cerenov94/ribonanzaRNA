import pandas as pd
import torch
import networkx as nx


def features_graph(seq,ss,bpps):
    pair_info = match_pair(ss)
    node_features = []
    edge_index = []
    edge_features = []

    paired_nodes = {}
    for j in range(len(seq)):
        add_base_node(node_features,seq[j])
        if j+1 < len(seq):
            add_edges_between_base_nodes(edge_index,edge_features,j,j+1)
        if pair_info[j] != -1:
            if pair_info[j] not in paired_nodes:
                paired_nodes[pair_info[j]] = [j]
            else:
                paired_nodes[pair_info[j]].append(j)
    for pair in paired_nodes.values():
        bpps_value = bpps[(bpps[0] == pair[0]) & (bpps[1] == pair[1])][2]
        add_edges_between_paired_nodes(edge_index, edge_features,pair[0], pair[1], bpps_value)


    node_features = torch.tensor(node_features,dtype=torch.float)
    edge_index = torch.LongTensor(edge_index).t().contiguous()
    edge_features = torch.tensor(edge_features,dtype=torch.float)

    return node_features,edge_index,edge_features


def add_edges(edge_index, edge_features, node1, node2, feature1, feature2):
    edge_index.append([node1, node2])
    edge_features.append(feature1)
    edge_index.append([node2, node1])
    edge_features.append(feature2)


def add_edges_between_base_nodes(edge_index, edge_features, node1, node2):
    edge_feature1 = [
        0, # is edge for paired nodes


        1, # forward edge: 1, backward edge: -1
        1, # bpps if edge is for paired nodes
    ]
    edge_feature2 = [
        0, # is edge for paired nodes


        -1, # forward edge: 1, backward edge: -1
        1, # bpps if edge is for paired nodes
    ]
    add_edges(edge_index, edge_features, node1, node2,
              edge_feature1, edge_feature2)


def add_edges_between_paired_nodes(edge_index, edge_features, node1, node2,
                                   bpps_value):
    edge_feature1 = [
        1, # is edge for paired nodes


        0, # forward edge: 1, backward edge: -1
        bpps_value, # bpps if edge is for paired nodes
    ]
    edge_feature2 = [
        1, # is edge for paired nodes


        0, # forward edge: 1, backward edge: -1
        bpps_value, # bpps if edge is for paired nodes
    ]
    add_edges(edge_index, edge_features, node1, node2,
              edge_feature1, edge_feature2)

def add_edges_between_codon_nodes(edge_index, edge_features, node1, node2):
    edge_feature1 = [
        0,  # is edge for paired nodes
        0,  # is edge between codon node and base node
        1,  # is edge between coden nodes
        1,  # forward edge: 1, backward edge: -1
        0,  # bpps if edge is for paired nodes
    ]
    edge_feature2 = [
        0,  # is edge for paired nodes
        0,  # is edge between codon node and base node
        1,  # is edge between coden nodes
        -1,  # forward edge: 1, backward edge: -1
        0,  # bpps if edge is for paired nodes
    ]
    add_edges(edge_index, edge_features, node1, node2,
              edge_feature1, edge_feature2)




def add_edges_between_codon_and_base_node(edge_index, edge_features,
                                          node1, node2):
    edge_feature1 = [
        0, # is edge for paired nodes
        1, # is edge between codon node and base node
        0, # is edge between coden nodes
        0, # forward edge: 1, backward edge: -1
        0, # bpps if edge is for paired nodes
    ]
    edge_feature2 = [
        0, # is edge for paired nodes
        1, # is edge between codon node and base node
        0, # is edge between coden nodes
        0, # forward edge: 1, backward edge: -1
        0, # bpps if edge is for paired nodes
    ]
    add_edges(edge_index, edge_features, node1, node2,
              edge_feature1, edge_feature2)



def match_pair(structure):
    pair = [-1] * len(structure)
    pair_no = -1

    pair_no_stack = []
    for i, c in enumerate(structure):
        if c == '(':
            pair_no += 1
            pair[i] = pair_no
            pair_no_stack.append(pair_no)
        elif c == ')':
            pair[i] = pair_no_stack.pop()
    return pair


def add_node(node_features, feature):
    node_features.append(feature)


def add_base_node(node_features, sequence):
    feature = [
        sequence == 'A',
        sequence == 'C',
        sequence == 'G',
        sequence == 'U',
    ]
    add_node(node_features, feature)

def ohe_seq(seq):
    node_features = []
    for j in range(len(seq)):
        add_base_node(node_features,seq[j])
    node_features = torch.tensor(node_features,dtype=torch.float)

    return node_features

def dotbracket_to_graph(dotbracket):
    G = nx.Graph()
    bases = []

    for i, c in enumerate(dotbracket):
        if c == '(':
            bases.append(i)
        elif c == ')':
            neighbor = bases.pop()
            G.add_edge(i, neighbor, edge_type='base_pair')
        elif c == '.':
            G.add_node(i)
        else:
            print("Input is not in dot-bracket notation!")
            return None

        if i > 0:
            G.add_edge(i, i - 1, edge_type='adjacent')
    return G


def extract_dinucleotide_features(sequence):
    dinucleotides = ['AA', 'AC', 'AG', 'AU', 'CA', 'CC', 'CG', 'CU', 'GA', 'GC', 'GG', 'GU', 'UA', 'UC', 'UG', 'UU']
    features = {}

    for dinucleotide in dinucleotides:
        features[dinucleotide] = sequence.count(dinucleotide)

    return features


def extract_trinucleotide_features(sequence):
    trinucleotides = ['AAA', 'AAC', 'AAG', 'AAU', 'ACA', 'ACC', 'ACG', 'ACU', 'AGA', 'AGC', 'AGG', 'AGU', 'AUA', 'AUC',
                      'AUG', 'AUU', 'CAA', 'CAC', 'CAG', 'CAU', 'CCA', 'CCC', 'CCG', 'CCU', 'CGA', 'CGC', 'CGG', 'CGU',
                      'CUA', 'CUC', 'CUG', 'CUU', 'GAA', 'GAC', 'GAG', 'GAU', 'GCA', 'GCC', 'GCG', 'GCU', 'GGA', 'GGC',
                      'GGG', 'GGU', 'GUA', 'GUC', 'GUG', 'GUU', 'UAA', 'UAC', 'UAG', 'UAU', 'UCA', 'UCC', 'UCG', 'UCU',
                      'UGA', 'UGC', 'UGG', 'UGU', 'UUA', 'UUC', 'UUG', 'UUU']
    features = {}

    for trinucleotide in trinucleotides:
        features[trinucleotide] = sequence.count(trinucleotide)

    return features


def edge_attributes(g):
    edge_attr = []
    for e in list(g.edges(data=True)):
        if e[2]['attr'] == 'phosphodiester_bond':
            edge_attr.append([1, 0, 0, 0])
        elif (e[2]['attr'] == 'base_pairing') and (e[2]['pairing_type'] == 'canonical'):
            edge_attr.append([0, 1, 1, 0])
        elif (e[2]['attr'] == 'base_pairing') and (e[2]['pairing_type'] == 'wobble'):
            edge_attr.append([0, 1, 0, 1])
    return edge_attr