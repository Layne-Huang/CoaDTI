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
import random
from torch_geometric.data import DataLoader
import re

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

    # GetSymbol: obtain the symbol of the atom
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
    # sequence = '-' + sequence + '='
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

def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes, device='cpu')  # [D,D]
    return y[labels]  # [N,D]



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


def pad(x,max_length):
    if len(x) > (max_length):
        x = x[:max_length]
    else:
        for i in range(max_length - len(x)):
            x = np.append(x, 0)
    return x

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def transfer_defacutdict(data):
    return defaultdict(int, data)


if __name__ == "__main__":

    # DATASET, radius, ngram = sys.argv[1:]
    DATASET = 'celegans' #'human' #"human" #"celegans"
    radius = 2
    ngram = 1
    element_symbol_list = ['Cl', 'N', 'S', 'F', 'C', 'O', 'H']

    radius, ngram = map(int, [radius, ngram])

    pre_split = False

    if pre_split:
        with open('../dataset/' + DATASET + '/original/train.txt', 'r') as f:
            data_list_train = f.read().strip().split('\n')  # strip: remove the space of the head and the tail
        with open('../dataset/' + DATASET + '/original/dev.txt', 'r') as f:
            data_list_dev = f.read().strip().split('\n')  
        with open('../dataset/' + DATASET + '/original/test.txt', 'r') as f:
            data_list_test = f.read().strip().split('\n')  
    else:
        with open('../dataset/' + DATASET + '/original/data.txt', 'r') as f:
            data_list = f.read().strip().split('\n') 

    """Exclude data contains '.' in the SMILES format."""

    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    # print(data_list[1].strip().split()[0])


    N = len(data_list)

    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    word_dict = defaultdict(lambda: len(word_dict))
    atom_m_dict = defaultdict(lambda: len(atom_m_dict))

    # dir_input = ('../dataset/' + '/dict/''radius' + str(
    #     radius) + '_ngram' + str(ngram) + '/')
    # fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    # word_dict = load_pickle(dir_input + 'word_dict.pickle')
    # bond_dict = load_pickle(dir_input + 'bond_dict.pickle')
    # edge_dict = load_pickle(dir_input + 'edge_dict.pickle')
    # atom_dict = load_pickle(dir_input + 'atom_dict.pickle')
    # fingerprint_dict = transfer_defacutdict(fingerprint_dict)
    # word_dict = transfer_defacutdict(word_dict)
    # bond_dict = transfer_defacutdict(bond_dict)
    # edge_dict = transfer_defacutdict(edge_dict)
    # atom_dict = transfer_defacutdict(atom_dict)

    Smiles, compounds, adjacencies, proteins, interactions = '', [], [], [], []
    data_geo_list = []
    tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)


    max_len = 4000
    min_len = 99999
    count = 0
    protein_list = []
    for data in data_list:
        smiles, sequence, interaction = data.strip().split()
        protein = split_sequence(sequence, ngram)
        protein_list.append(protein)

    num_words = len(word_dict)


    for no, data in enumerate(tqdm((data_list))):

        smiles, _, interaction = data.strip().split()
        sequence = protein_list[no]

        if len(sequence)>max_len:
            sequence = sequence[:max_len]

        count+=1

        Smiles += smiles + '\n'

        if max_len<len(sequence):
            max_len = len(sequence)
        if min_len>len(sequence):
            min_len = len(sequence)

        # Add hydrogens

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Consider hydrogens.
        atoms = create_atoms(mol)
        atoms_m = GetVertMat(mol,element_symbol_list)
        atoms_m = torch.tensor(atoms_m, dtype=torch.float)
        i_jbond_dict = create_ijbonddict(mol)
        fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)
        fingerprints = torch.tensor(fingerprints, dtype=torch.long).unsqueeze(1)

        compounds.append(fingerprints)

        adjacency = create_adjacency(mol)
        adjacencies.append(adjacency)

        protein = sequence

        adjacency = coo_matrix(adjacency)
        edge_index = [adjacency.row, adjacency.col]
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        interactions.append(np.array([float(interaction)]))

        interaction = torch.tensor(int(interaction))
        protein = torch.tensor(protein, dtype=torch.long)

        data = Data(x=fingerprints, edge_index=edge_index, y=interaction, protein=protein)

        data_geo_list.append(data)

    dataset = random_shuffle(data_geo_list, 1234)
    dataset_train, dataset_ = split_dataset(data_geo_list, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)

    dir_input = ('../dataset/' + DATASET + '/input/final/'
                 'radius' + str(radius) + '_ngram' + str(ngram) + '/')
    os.makedirs(dir_input, exist_ok=True)


    # no-padding
    # torch.save(dataset_train, dir_input + 'drug-target_train_3' + ".pt")
    # torch.save(dataset_dev, dir_input + 'drug-target_dev_3' + ".pt")
    # torch.save(dataset_test, dir_input + 'drug-target_test_3' + ".pt")

    # batch
    # torch.save(dataset_train, dir_input + 'drug-target_train_batch_3' + ".pt")
    # torch.save(dataset_dev, dir_input + 'drug-target_dev_batch_3' + ".pt")
    # torch.save(dataset_test, dir_input + 'drug-target_test_batch_3' + ".pt")

    # no-radius+batch
    # torch.save(dataset_train, dir_input + 'drug-target_train_batch_nrad_3' + ".pt")
    # torch.save(dataset_dev, dir_input + 'drug-target_dev_batch_nrad_3' + ".pt")
    # torch.save(dataset_test, dir_input + 'drug-target_test_batch_nrad_3' + ".pt")

    torch.save(dataset_train, dir_input + 'drug-target_train_{}.pt'.format(max_len))
    torch.save(dataset_dev, dir_input + 'drug-target_dev_{}.pt'.format(max_len))
    torch.save(dataset_test, dir_input + 'drug-target_test_{}.pt'.format(max_len))


    # dump_dictionary(word_dict, dir_input + 'word_dict.pickle')
    # dump_dictionary(fingerprint_dict, dir_input + 'fingerprint_dict.pickle')

    print('The preprocess of ' + DATASET + ' dataset has finished!')
