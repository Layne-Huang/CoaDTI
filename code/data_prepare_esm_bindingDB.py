from collections import defaultdict
import os
import pickle
import sys
import torch
import numpy as np
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
from rdkit import Chem
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, XLNetTokenizer
from torch_geometric.data import DataLoader
import re
import random

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def split_words(str):
    s = ''
    for i,x in enumerate(str):
        if i!=len(str):
            s+=x+" "
        else:
            s+=x
    return s

def tarlor_side(sequence, length):
    if len(sequence)>length:
        tailor_n = int((len(sequence)-length)/2)
        sequence = sequence[tailor_n-1:tailor_n-1+length]

    return sequence

def insert_sep(sequence, length):
    if len(sequence)>length:
        number = len(sequence)//length
        for i in range(number):
            sequence.insert(length*(i+1)+i, '[SEP]')
    return str(sequence)

def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""

    # GetSymbol 获取分子的元素符号
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()

        atoms[i] = (atoms[i], 'aromatic')
    # print(atom_dict["H"])
    # print(atoms)

    # turn it into index
    # print(atoms)
    atoms = [atom_dict[a] for a in atoms]

    return np.array(atoms)

def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    # print(sequence)
    words = [word_dict[sequence[i:i+ngram]]
             for i in range(len(sequence)-ngram+1)]
    return np.array(words)


def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        print(x,type(x))
        raise Exception("input {0} not in allowable set{1}:".format(
                x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def GetVertMat(mol, element_symbol_list):
    # fingureprints = []
    V = []
    # for atoms in mol:
    # for a in mol.GetAromaticAtoms():
    #     i = a.GetIdx()
    #     atoms[i] = (atoms[i], 'aromatic')
    for i, a in enumerate(mol.GetAtoms()):
        v = []
        # print(a.GetSymbol())
        index = a.GetIdx()
        v.append(index)
        v.append(a.GetMass())
        V.append(v)
        # fingureprints.append(V)
    # V = [atom_m_dict[v] for v in V]
    return np.array(V)

# def max_len(sequence):
#     m = 0
#
#     for s in sequence:
#         if len(s)>m:
#             m=len(s)
#     return m

def split_data(data):
    length = len(data)
    new_data = []
    negative_counts = 0
    positive_counts = 0
    for x in data:
        smiles, sequence, interaction = x.strip().split()
        if int(interaction)==0 and negative_counts<length//40:
            new_data.append(x)
            negative_counts+=1
            continue
        if int(interaction)==1 and positive_counts<length//40:
            new_data.append(x)
            positive_counts+=1
            continue
        if negative_counts>length//40 and positive_counts>length//40:
            break
    random.shuffle(new_data)
    return new_data

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def transfer_defacutdict(data):
    return defaultdict(int, data)

def construct(data, tokenizer):
    max_len = 4000
    min_len = 99999
    count = 0
    data_geo_list = []
    Smiles, compounds, adjacencies, proteins, interactions = '', [], [], [], []

    for no, data in enumerate(tqdm(data)):
        smiles, sequence, interaction = data.strip().split()
        interaction = torch.tensor(int(interaction))
        # print(sequence)
        # print(smiles)
        if interaction==torch.tensor(0):
            count+=1
        #     print('start:{}'.format(interaction))
        #     interaction = torch.tensor(int(interaction))
        #     print(interaction)


        Smiles += smiles + '\n'

        if len(sequence)>max_len:
            sequence = sequence[:max_len]

        # Add hydrogens
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Consider hydrogens.

        atoms = create_atoms(mol)

        atoms_m = GetVertMat(mol,element_symbol_list)
        atoms_m = torch.tensor(atoms_m, dtype=torch.float)

        i_jbond_dict = create_ijbonddict(mol)

        # print(atoms.shape)
        fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
        fingerprints = torch.tensor(fingerprints, dtype=torch.long).unsqueeze(1)
        compounds.append(fingerprints)
        adjacency = create_adjacency(mol)
        adjacencies.append(adjacency)

        words = split_sequence(sequence, ngram)



        # protein = split_words(tarlor_side(sequence,tailor_length))

        # protein = re.sub(r"[UZOB]", "X", protein)
        # protein = tokenizer(protein, padding="longest", truncation=True, return_tensors='pt')

        adjacency = coo_matrix(adjacency)
        edge_index = [adjacency.row, adjacency.col]
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        proteins.append(words)

        # interactions.append(np.array([float(interaction)]))

        # if int(interaction) == 0:
        #     print(interaction)
        # if int(interaction) == 1:
        #     print(interaction)

        words = torch.tensor(words, dtype=torch.long)

        data = Data(x=fingerprints, edge_index=edge_index, y=interaction, protein=sequence)

        data_geo_list.append(data)
    print(count)
    return data_geo_list

if __name__ == "__main__":

    # DATASET, radius, ngram = sys.argv[1:]
    DATASET = "binding_DB"
    radius = 2
    ngram = 1
    element_symbol_list = ['Cl', 'N', 'S', 'F', 'C', 'O', 'H']

    radius, ngram = map(int, [radius, ngram])

    pre_split = True

    if pre_split:
        with open('../dataset/' + DATASET + '/original/train.txt', 'r') as f:
            data_list_train = f.read().strip().split('\n')  # strip去除首尾空格
        with open('../dataset/' + DATASET + '/original/dev.txt', 'r') as f:
            data_list_dev = f.read().strip().split('\n')  # strip去除首尾空格
        with open('../dataset/' + DATASET + '/original/test.txt', 'r') as f:
            data_list_test = f.read().strip().split('\n')  # strip去除首尾空格
    else:
        with open('../dataset/' + DATASET + '/original/data.txt', 'r') as f:
            data_list = f.read().strip().split('\n') # strip去除首尾空格

    """Exclude data contains '.' in the SMILES format."""

    data_list_sdsp = np.load('../dataset/binding_DB/input/split_test_data/seen_drug_seen_protein.npy')
    data_list_sdusp = np.load('../dataset/binding_DB/input/split_test_data/seen_drug_unseen_protein.npy')
    data_list_usdsp = np.load('../dataset/binding_DB/input/split_test_data/unseen_drug_seen_protein.npy')
    data_list_usdusp = np.load('../dataset/binding_DB/input/split_test_data/unseen_drug_unseen_protein.npy')

    data_list_train = [d for d in data_list_train if '.' not in d.strip().split()[0]]
    # count=0
    # for x in tqdm(data_list_train):
    #     smiles, sequence, interaction = x.strip().split()
    #     if interaction=='0':
    #         count+=1
    #         print(interaction)
    # print(count)
    # exit()
    data_list_dev = [d for d in data_list_dev if '.' not in d.strip().split()[0]]
    data_list_test = [d for d in data_list_test if '.' not in d.strip().split()[0]]

    data_list_sdsp = [d for d in data_list_sdsp if '.' not in d.strip().split()[0]]
    data_list_sdusp = [d for d in data_list_sdusp if '.' not in d.strip().split()[0]]
    data_list_usdsp = [d for d in data_list_usdsp if '.' not in d.strip().split()[0]]
    data_list_usdusp = [d for d in data_list_usdusp if '.' not in d.strip().split()[0]]

    # print(data_list[1].strip().split()[0])

    # atom_dict = defaultdict(lambda: len(atom_dict))
    # bond_dict = defaultdict(lambda: len(bond_dict))
    # fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    # edge_dict = defaultdict(lambda: len(edge_dict))
    # word_dict = defaultdict(lambda: len(word_dict))
    # atom_m_dict = defaultdict(lambda: len(atom_m_dict))

    dir_input = ('../dataset/' + '/dict/'+'radius' + str(
        radius) + '_ngram' + str(ngram) + '/')
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'word_dict.pickle')
    bond_dict = load_pickle(dir_input + 'bond_dict.pickle')
    edge_dict = load_pickle(dir_input + 'edge_dict.pickle')
    atom_dict = load_pickle(dir_input + 'atom_dict.pickle')
    fingerprint_dict = transfer_defacutdict(fingerprint_dict)
    word_dict = transfer_defacutdict(word_dict)
    bond_dict = transfer_defacutdict(bond_dict)
    edge_dict = transfer_defacutdict(edge_dict)
    atom_dict = transfer_defacutdict(atom_dict)



    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

    dataset_train = construct(data_list_train, tokenizer)
    dataset_dev = construct(data_list_dev, tokenizer)
    dataset_test = construct(data_list_test, tokenizer)

    # dataset_sdsp = construct(data_list_sdsp, tokenizer)
    # dataset_sdusp = construct(data_list_sdusp, tokenizer)
    # dataset_usdsp = construct(data_list_usdsp, tokenizer)
    # dataset_usdusp = construct(data_list_usdusp, tokenizer)



    dir_input = ('../dataset/' + DATASET + '/input/'
                 'radius' + str(radius) + '_ngram' + str(ngram) + '/final/')

    os.makedirs(dir_input, exist_ok=True)

    # dir_input = ('../dataset/' + DATASET + '/input/split_test_data/')

    # torch.save(dataset_test, dir_input + 'drug-target_test_esmpro_4000_o' + ".pt")
    # torch.save(dataset_sdsp, dir_input + 'drug-target_sdsp_esmpro_4000' + ".pt") # Only use 20% of the raw data
    # torch.save(dataset_sdusp, dir_input + 'drug-target_sdusp_esmpro_4000' + ".pt")
    # torch.save(dataset_usdsp, dir_input + 'drug-target_usdsp_esmpro_4000' + ".pt")
    # torch.save(dataset_usdusp, dir_input + 'drug-target_usdusp_esmpro_4000' + ".pt")
    torch.save(dataset_train, dir_input + 'drug-target_train_esmpro_4000' + ".pt") # Only use 20% of the raw data
    torch.save(dataset_dev, dir_input + 'drug-target_dev_esmpro_4000' + ".pt")
    torch.save(dataset_test, dir_input + 'drug-target_test_esmpro_4000' + ".pt")

    # with open(dir_input + 'Smiles.txt', 'w') as f:
    #     f.write(Smiles)
    # np.save(dir_input + 'compounds', compounds)
    # np.save(dir_input + 'adjacencies', adjacencies)
    # np.save(dir_input + 'proteins', proteins)
    # np.save(dir_input + 'interactions', interactions)
    print(len(fingerprint_dict))
    dir_input = ('../dataset/' + '/dict/''radius' + str(
        radius) + '_ngram' + str(ngram) + '/')
    dump_dictionary(fingerprint_dict, dir_input + 'fingerprint_dict_radius0.pickle')
    dump_dictionary(word_dict, dir_input + 'word_dict.pickle')
    dump_dictionary(atom_dict, dir_input + 'atom_dict.pickle')
    dump_dictionary(edge_dict, dir_input + 'edge_dict.pickle')
    dump_dictionary(bond_dict, dir_input + 'bond_dict.pickle')

    print('The preprocess of ' + DATASET + ' dataset has finished!')

