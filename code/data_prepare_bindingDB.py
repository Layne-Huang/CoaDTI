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
import pandas as pd
import csv

def random_shuffle(dataset, seed):
    random.seed(seed) #2021 2345 1234
    random.shuffle(dataset)
    return dataset

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

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def transfer_defacutdict(data):
    return defaultdict(int, data)

def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)


def split_sequence(sequence, ngram):
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

def pad(x,max_length):
    if len(x) > (max_length):
        x = x[:max_length]
    else:
        for i in range(max_length - len(x)):
            x = np.append(x, 0)
    return x

def construct(data_list, tokenizer, max_len):
    min_len = 99999
    count = 0
    data_geo_list = []
    Smiles, compounds, adjacencies, proteins, interactions = '', [], [], [], []
    protein_list = []
    # for data in data_list:
    #     # smiles, sequence, interaction = data.strip().split()
    #     smiles, sequence, interaction = data[1], data[2], data[0]
    #     protein = split_sequence(sequence, ngram)
    #     protein_list.append(protein)

    for no, data in enumerate(tqdm(data_list)):

        smiles, sequence, interaction = data.strip().split()

        if len(sequence) > max_len:
            sequence = sequence[:max_len]
        count += 1

        Smiles += smiles + '\n'


        # Add hydrogens

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Consider hydrogens.

        atoms = create_atoms(mol)

        atoms_m = GetVertMat(mol, element_symbol_list)
        atoms_m = torch.tensor(atoms_m, dtype=torch.float)

        i_jbond_dict = create_ijbonddict(mol)

        fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)

        fingerprints = torch.tensor(fingerprints, dtype=torch.long).unsqueeze(1)

        compounds.append(fingerprints)

        adjacency = create_adjacency(mol)
        adjacencies.append(adjacency)

        protein = split_sequence(sequence, ngram)
        adjacency = coo_matrix(adjacency)
        edge_index = [adjacency.row, adjacency.col]
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        interaction = torch.tensor(int(interaction))
        protein = torch.tensor(protein, dtype=torch.long)

        # print(fingerprints)
        # print(edge_index)
        # print(protein)
        # exit()
        data = Data(x=fingerprints, edge_index=edge_index, y=interaction, protein=protein)

        data_geo_list.append(data)
    print(count)
    return data_geo_list

if __name__ == "__main__":

    # DATASET, radius, ngram = sys.argv[1:]
    DATASET = "binding_DB"
    radius = 2
    ngram = 1
    max_len = 1000
    element_symbol_list = ['Cl', 'N', 'S', 'F', 'C', 'O', 'H']

    radius, ngram = map(int, [radius, ngram])

    pre_split = True

    '''Read the data'''

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

    # with open('../dataset/' + DATASET + '/train.csv', 'r') as f:
    #     data_list_train = list(csv.reader(f))[1:]
    #     # data_list_train = [d[5:] for d in data_list_train]
    #     data_list_train = [d[1:4] for d in data_list_train]
    # with open('../dataset/' + DATASET + '/val.csv', 'r') as f:
    #     data_list_dev = list(csv.reader(f))[1:]
    #     # data_list_dev = [d[5:] for d in data_list_dev]
    #     data_list_dev = [d[1:4] for d in data_list_dev]
    # with open('../dataset/' + DATASET + '/test.csv', 'r') as f:
    #     data_list_test = list(csv.reader(f))[1:]
    #     # data_list_test = [d[5:] for d in data_list_test]
    #     data_list_test = [d[1:4] for d in data_list_test]


    """Exclude data contains '.' in the SMILES format."""

    '''Celegans dataset/human/binding_DB'''
    # data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    data_list_train = [d for d in data_list_train if '.' not in d.strip().split()[0]]
    data_list_dev = [d for d in data_list_dev if '.' not in d.strip().split()[0]]
    data_list_test = [d for d in data_list_test if '.' not in d.strip().split()[0]]

    '''BIOSNAP'''
    # data_list_train = [d for d in data_list_train if '.' not in d[1]]
    # data_list_dev = [d for d in data_list_dev if '.' not in d[1]]
    # data_list_test = [d for d in data_list_test if '.' not in d[1]]
    # data_list_train = [d for d in data_list_train if '.' not in d[0]]
    # data_list_dev = [d for d in data_list_dev if '.' not in d[0]]
    # data_list_test = [d for d in data_list_test if '.' not in d[0]]
    # print(data_list_train[0][0][0])
    # exit()
    # print(data_list[1].strip().split()[0])

    # atom_dict = defaultdict(lambda: len(atom_dict))
    # bond_dict = defaultdict(lambda: len(bond_dict))
    # fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    # edge_dict = defaultdict(lambda: len(edge_dict))
    # word_dict = defaultdict(lambda: len(word_dict))
    # atom_m_dict = defaultdict(lambda: len(atom_m_dict))

    dir_input = ('../dataset/' + '/dict/''radius' + str(
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

    dataset_train = construct(data_list_train, tokenizer, max_len)
    dataset_dev = construct(data_list_dev, tokenizer, max_len)
    dataset_test = construct(data_list_test, tokenizer, max_len)

    # data_list = construct(data_list, tokenizer, max_len)
    # dataset = random_shuffle(data_list, 2021)
    # dataset_train, dataset_ = split_dataset(data_list, 0.8)
    # dataset_dev, dataset_test = split_dataset(dataset_, 0.5)

    dir_input = ('../dataset/' + DATASET + '/input/'
                 'radius' + str(radius) + '_ngram' + str(ngram) + '/final/')
    os.makedirs(dir_input, exist_ok=True)

    torch.save(dataset_train, dir_input + 'drug-target_train_{}.pt'.format(max_len)) # Only use 20% of the raw data
    torch.save(dataset_dev, dir_input + 'drug-target_dev_{}.pt'.format(max_len))
    torch.save(dataset_test, dir_input + 'drug-target_test_{}.pt'.format(max_len))

    # with open(dir_input + 'Smiles.txt', 'w') as f:
    #     f.write(Smiles)
    # np.save(dir_input + 'compounds', compounds)
    # np.save(dir_input + 'adjacencies', adjacencies)
    # np.save(dir_input + 'proteins', proteins)
    # np.save(dir_input + 'interactions', interactions)
    # print(len(fingerprint_dict))
    print(len(fingerprint_dict))
    dump_dictionary(fingerprint_dict, dir_input + 'fingerprint_dict.pickle')
    dump_dictionary(word_dict, dir_input + 'word_dict.pickle')

    print('The preprocess of ' + DATASET + ' dataset has finished!')

