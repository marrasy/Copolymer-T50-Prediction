import csv
import functools
import os.path as osp
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data, DataLoader, Batch
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter


x_map = {
    'atomic_num': list(range(0, 119)),
    'degree': list(range(0, 11)),
    'formal_charge': list(range(-5, 7)),
    'num_hs': list(range(0, 9)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
    'levels': ['head', 'gating', 'atom'],
}


e_map = {

    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
    'levels': ['h2g', 'g2g', 'g2a', 'a2a'],
}


def onehot(feature_list, cur_feature):
    assert cur_feature in feature_list
    vector = [0] * len(feature_list)
    index = feature_list.index(cur_feature)
    vector[index] = 1
    return vector

def onehot_zero(feature_list):
    vector = [0] * len(feature_list)
    return vector


def process_smiles(smiles, form_ring, has_H):
    """Form a ring molecule for monomer."""
    mol = Chem.MolFromSmiles(smiles)
    if has_H:
        mol = Chem.AddHs(mol)
    Chem.SanitizeMol(mol)
    return mol


class MultiDataset(Dataset):
    """Combine two dataset together."""
    def __init__(self, exp_data, sim_data):
        assert len(exp_data) <= len(sim_data)
        self.exp_data = exp_data
        self.sim_data = sim_data

    def __len__(self):
        return len(self.sim_data)

    def __getitem__(self, idx):
        sim_d = self.sim_data[idx]
        exp_d = self.exp_data[idx % len(self.exp_data)]
        return exp_d, sim_d


class PolymerDataset(Dataset):


    def merge_graphs(self, graph_list, frac_list, tg):
        x, edge_indices, edge_attrs = [], [], []
        tg_float=float(tg)
        # Start with "head" node
        xs = []
        tmpx = []
        tmpx += onehot(x_map['levels'], str("head"))
        tmpx += onehot_zero(x_map['atomic_num'])
        tmpx += onehot_zero(x_map['degree'])
        tmpx += onehot_zero(x_map['formal_charge'])
        tmpx += onehot_zero(x_map['num_hs'])
        tmpx += onehot_zero(x_map['hybridization'])
        tmpx += onehot_zero(x_map['is_aromatic'])
        tmpx += onehot_zero(x_map['is_in_ring'])
        tmpx += [tg_float]
        tmpx += [0]
        xs.append(tmpx)

        #gating node的建立
        for frac_value in frac_list:
            tmpx = []
            tmpx += onehot(x_map['levels'], str("gating"))
            tmpx += onehot_zero(x_map['atomic_num'])
            tmpx += onehot_zero(x_map['degree'])
            tmpx += onehot_zero(x_map['formal_charge'])
            tmpx += onehot_zero(x_map['num_hs'])
            tmpx += onehot_zero(x_map['hybridization'])
            tmpx += onehot_zero(x_map['is_aromatic'])
            tmpx += onehot_zero(x_map['is_in_ring'])
            tmpx += [0]
            tmpx += [frac_value]
            xs.append(tmpx)

        add_x=torch.tensor(xs).to(torch.float)
        x.extend(add_x)


        for i in range(1, len(x)):
            edge_indices += [[i, 0],[0, i]]  # Connect head node (0) to the gating
            tmpe = []
            tmpe += onehot(e_map['levels'], str("h2g"))
            tmpe += onehot_zero(e_map['bond_type'])
            tmpe += onehot_zero(e_map['stereo'])
            tmpe += onehot_zero(e_map['is_conjugated'])
            tmpe += [0] 
            tmpe_tensor = torch.tensor(tmpe).to(torch.float)
            edge_attrs.append(tmpe_tensor)
            edge_attrs.append(tmpe_tensor)



        for i in range(1, len(x)-1):
            for j in range(i+1,len(x)):
                edge_indices += [[i, j],[j, i]]  # Connect head node (0) to the gating
                tmpe = []
                tmpe += onehot(e_map['levels'], str("g2g"))
                tmpe += onehot_zero(e_map['bond_type'])
                tmpe += onehot_zero(e_map['stereo'])
                tmpe += onehot_zero(e_map['is_conjugated'])
                tmpe += [0] 
                tmpe_tensor = torch.tensor(tmpe).to(torch.float)
                edge_attrs.append(tmpe_tensor)
                edge_attrs.append(tmpe_tensor)
        
        node_counter = len(x)  # the first node of the first graph will have this index

        # Adding edges between gating nodes and nodes of corresponding subgraphs
        for idx, graph in enumerate(graph_list):
            num_nodes_graph = len(graph.x)  # number of nodes in this graph
            x.extend(graph.x)
        
            # adding edges between the gating node and nodes of the graph
            for node_idx_graph in range(num_nodes_graph):
                edge_indices += [[1 + idx, node_counter + node_idx_graph], 
                                [node_counter + node_idx_graph, 1 + idx]]
                tmpe = []
                tmpe += onehot(e_map['levels'], str("g2a"))
                tmpe += onehot_zero(e_map['bond_type'])
                tmpe += onehot_zero(e_map['stereo'])
                tmpe += onehot_zero(e_map['is_conjugated'])
                tmpe += [0] 
                tmpe_tensor = torch.tensor(tmpe).to(torch.float)
                edge_attrs.append(tmpe_tensor)
                edge_attrs.append(tmpe_tensor)

            # adjusting the edge indices in the graph
            for src, dest in graph.edge_index.t().tolist():
                edge_indices.append([src + node_counter, dest + node_counter])

            edge_attrs.extend(graph.edge_attr)

            node_counter += num_nodes_graph  # update the counter

        x = torch.stack(x)
        edge_index = torch.tensor(edge_indices)
        edge_index = edge_index.t().to(torch.long).view(2, -1)
        edge_attr = torch.stack(edge_attrs)

        merged_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        return merged_graph

    

    def __init__(self, root_dir, type, split, total_split=10, log10=True,
                 form_ring=True, has_H=True, size_limit=None, pred_csv=None):
        assert split < 10
        csv_files = []
        if type == 'train':
            for i in range(total_split):
                if i != split:
                    csv_files.append(osp.join(root_dir, 'cv_{}.csv'.format(i)))
        elif type == 'val':
            csv_files.append(osp.join(root_dir, 'cv_{}.csv'.format(split)))
        elif type == 'test':
            csv_files.append(osp.join(root_dir, 'test.csv'))
        elif type == 'pred':
            if pred_csv is None:
                pred_csv = osp.join(root_dir, 'pred.csv')
            csv_files.append(pred_csv)
        self.raw_data = []
        for csv_file in csv_files:
            with open(csv_file) as f:
                rows = [row for row in csv.reader(f)]
            self.raw_data += rows
        np.random.seed(123)
        np.random.shuffle(self.raw_data)
        if size_limit is not None:
            self.raw_data = self.raw_data[:size_limit]
        self.log10 = log10
        self.form_ring = form_ring
        self.has_H = has_H

        print('Type {} csvs {}'.format(type, [c.split('/')[-1] for c in csv_files]))

    def __len__(self):
        return len(self.raw_data)
    
    @staticmethod
    def get_frac_percentages(frac):
        counter = Counter(frac)
        total_count = sum(counter.values())
        percentages = [count / total_count for count in counter.values()]
        return percentages

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, idx):
        poly_id, smiles_full, target_origin, frac, tg = self.raw_data[idx]
        smiles_list = smiles_full.split('.')
        data_list = []

        frac_list= self.get_frac_percentages(frac)


        for smiles in smiles_list:
            mol = process_smiles(smiles, form_ring=self.form_ring, has_H=self.has_H)

            target = float(target_origin)
            if self.log10:
                target = np.log10(target)
            target = torch.tensor(target).float()

            xs = []
            for atom in mol.GetAtoms():
                x = []
                x += onehot(x_map['levels'], str("atom"))
                x += onehot(x_map['atomic_num'], atom.GetAtomicNum())
                x += onehot(x_map['degree'], atom.GetTotalDegree())
                x += onehot(x_map['formal_charge'], atom.GetFormalCharge())
                x += onehot(x_map['num_hs'], atom.GetTotalNumHs())
                x += onehot(x_map['hybridization'], str(atom.GetHybridization()))
                x += onehot(x_map['is_aromatic'], atom.GetIsAromatic())
                x += onehot(x_map['is_in_ring'], atom.IsInRing())
                x += [0]
                x += [0]
                xs.append(x)

            x = torch.tensor(xs).to(torch.float)

            edge_indices, edge_attrs = [], []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                e = []
                e += onehot(e_map['levels'], str("a2a"))
                e += onehot(e_map['bond_type'], str(bond.GetBondType()))
                e += onehot(e_map['stereo'], str(bond.GetStereo()))
                e += onehot(e_map['is_conjugated'], bond.GetIsConjugated())
                e += [0]

                edge_indices += [[i, j], [j, i]]
                edge_attrs += [e, e]

            edge_index = torch.tensor(edge_indices)
            edge_index = edge_index.t().to(torch.long).view(2, -1)
            edge_attr = torch.tensor(edge_attrs).to(torch.float)

            # Sort indices.
            if edge_index.numel() > 0:
                perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
                edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=target, smiles=smiles, poly_id=poly_id, frac=frac,tg=tg)
            data_list.append(data)

        target_origin = float(target_origin)
        if self.log10:
            target_origin = np.log10(target_origin)
        target_origin = torch.tensor(target_origin).float()
        combined_data = self.merge_graphs(data_list, frac_list, tg)
        combined_data.y = target_origin
        combined_data.smiles = smiles_full
        combined_data.poly_id = poly_id
        combined_data.frac = frac
        combined_data.tg = tg

        return combined_data
