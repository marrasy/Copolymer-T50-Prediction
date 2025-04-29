#smiles=['C=CP(=O)(O)O','C=CC(=O)N','C=CC(=O)NC(CO)(CO)CO','CC(C)(CS(=O)(=O)O)NC(=O)C=C','CC(=C)C(=O)OCCOP(=O)([O-])OCC[N+](C)(C)C','C[N+](C)(C)CCOC(=O)C=C[Cl-]']

import numpy as np
import torch
import dgl
from dgl import DGLGraph
from rdkit import Chem
import networkx as nx
import matplotlib.pyplot as plt

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def get_atom_features(atom):
    possible_atom = ['C', 'N', 'O', 'Cl', 'P', 'S']
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atom)
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1,0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(),
                                           [Chem.rdchem.HybridizationType.SP,
                                            Chem.rdchem.HybridizationType.SP2,
                                            Chem.rdchem.HybridizationType.SP3,
                                            Chem.rdchem.HybridizationType.SP3D])
    return np.array(atom_features,dtype=int)

def get_bond_features(bond):
    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE, bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE, bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return np.array(bond_feats, dtype=int)

def smiles_to_dgl_graph(smiles):
    G = dgl.graph([])
    G = dgl.add_self_loop(G)
    molecule = Chem.MolFromSmiles(smiles)
    G.add_nodes(molecule.GetNumAtoms())

    node_features = []
    edge_features = []

    for i in range(molecule.GetNumAtoms()):
        atom_i = molecule.GetAtomWithIdx(i)
        atom_i_features = get_atom_features(atom_i)
        node_features.append(atom_i_features)

        for j in range(molecule.GetNumAtoms()):
            bond_ij = molecule.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                G.add_edges(i, j)
                bond_features_ij = get_bond_features(bond_ij)
                edge_features.append(bond_features_ij)

    G.ndata['x'] = torch.from_numpy(np.array(node_features))
    G.edata['w'] = torch.from_numpy(np.array(edge_features))
    #输出画图
    #
    return G

def loadGraphData(smiles_list):
    all_graphs = []
    for smiles in smiles_list:
        graph = smiles_to_dgl_graph(smiles)
        all_graphs.append(graph)
    return all_graphs

smiles_list_example = ['C=CP(=O)(O)O', 'C=CC(=O)N', 'C=CC(=O)NC(CO)(CO)CO', 'CC(C)(CS(=O)(=O)O)NC(=O)C=C',
                           'CC(=C)C(=O)OCCOP(=O)([O-])OCC[N+](C)(C)C', 'C[N+](C)(C)CCOC(=O)C=C.[Cl-]']
all_graphs_example = loadGraphData(smiles_list_example)

for i, graph in enumerate(all_graphs_example):
    #大体输出
    print(f"Graph {i+1} - Number of nodes: {graph.number_of_nodes()}, Number of edges: {graph.number_of_edges()}")
    # 输出每个节点的特征向量
    for node, feat in enumerate(graph.ndata['x']):
        print(f"Node {node} Feature Vector: {feat.numpy()}")
    # 输出每条边的特征向量
    for edge, feat in enumerate(graph.edata['w']):
        src, dst = graph.find_edges(edge)
        print(f"Edge {src.item()} -> {dst.item()} Feature Vector: {feat.numpy()}")
    # 画图
    # nx_G = graph.to_networkx().to_undirected()
    # pos = nx.kamada_kawai_layout(nx_G)
    # nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    # plt.show()
    # print(graph.idtype)
def loadSpecificGraph():
    smiles_list_example = ['C=CP(=O)(O)O', 'C=CC(=O)N', 'C=CC(=O)NC(CO)(CO)CO', 'CC(C)(CS(=O)(=O)O)NC(=O)C=C',
                           'CC(=C)C(=O)OCCOP(=O)([O-])OCC[N+](C)(C)C', 'C[N+](C)(C)CCOC(=O)C=C.[Cl-]']
    all_graphs_example = loadGraphData(smiles_list_example)
    return all_graphs_example
