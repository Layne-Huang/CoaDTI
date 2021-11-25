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
        # tailor_n = int((len(sequence)-length)/2)
        # sequence = sequence[tailor_n-1:tailor_n-1+length]
        sequence = sequence[:length]

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

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def transfer_defacutdict(data):
    return defaultdict(int, data)

if __name__ == "__main__":

    # DATASET, radius, ngram = sys.argv[1:]
    DATASET = "human"
    radius = 2
    ngram = 1
    element_symbol_list = ['Cl', 'N', 'S', 'F', 'C', 'O', 'H']

    radius, ngram = map(int, [radius, ngram])

    pre_split = False

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
    # dir_input = ('../dataset/' + DATASET + '/input/'
    #                                        'radius' + str(radius) + '_ngram' + str(ngram) + '/final/')
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
    # tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
    # print(max_len(proteins))
    # exit()
    max_len = 0
    min_len = 99999
    count  = 0
    for no, data in enumerate(tqdm((data_list))):
        smiles, sequence, interaction = data.strip().split()

        count+=1
        Smiles += smiles + '\n'

        if max_len<len(sequence):
            max_len = len(sequence)
        if min_len>len(sequence):
            min_len = len(sequence)

        max_length = 4000
        if len(sequence)>max_length:
            sequence = sequence[:max_length]

        # Add hydrogens

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Consider hydrogens.
        # print(mol)

        atoms = create_atoms(mol)

        atoms_m = GetVertMat(mol,element_symbol_list)
        atoms_m = torch.tensor(atoms_m, dtype=torch.float)
        # print(atoms_m)
        # print(len(atom_dict))
        # print(atoms_m.shape)

        i_jbond_dict = create_ijbonddict(mol)

        # print(atoms.shape)
        fingerprints = extract_fingerprints(atoms, i_jbond_dict, radius)

        #
        fingerprints = torch.tensor(fingerprints, dtype=torch.long).unsqueeze(1)
        # fingerprints = GetVertMat(atoms, mol, element_symbol_list)

        # print(fingerprints.shape)
        # print(fingerprints)
        # print(len(fingerprint_dict))
        # exit()

        compounds.append(fingerprints)


        adjacency = create_adjacency(mol)
        adjacencies.append(adjacency)

        words = split_sequence(sequence, ngram)
        # print(sequence[:100])
        # print(sequence)
        # print(sequence[:1000])
        # print(sequence[:90])
        # exit()

        tailor_length = 4000
        # if len(sequence)>tailor_length and len(sequence)<700:
        #     print(len(sequence))
        #     print(sequence)
        #     print(tarlor_side(sequence,tailor_length))
        #     exit()

        # protein = split_words(tarlor_side(sequence,tailor_length))#cut

        # protein = split_words(sequence)

        # sequence = insert_sep(list(sequence), tailor_length)

        # protein = split_words(sequence)

        # protein = split_words(sequence)

        # protein = re.sub(r"[UZOB]", "X", protein)
        # protein = tokenizer(protein, padding="longest", truncation=True, return_tensors='pt')

        # print("sequence:\t",sequence)
        # print("words:\t", words)
        # print("compound:\t", fingerprints)
        # print(fingerprints.shape)
        adjacency = coo_matrix(adjacency)
        edge_index = [adjacency.row, adjacency.col]
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        # print("edge_index:\t", edge_index)
        # print(adjacency.shape)
        # exit()
        proteins.append(words)

        interactions.append(np.array([float(interaction)]))

        interaction = torch.tensor(int(interaction))
        words = torch.tensor(words, dtype=torch.long)

        data = Data(x=fingerprints, edge_index=edge_index, y=interaction, protein=sequence)
        # print(data)
        # print('....')
        # exit()
        data_geo_list.append(data)
    # print(max_len)
    # print(min_len)
    # exit()
    # print(interactions)
    # exit()
    dataset = random_shuffle(data_geo_list, 1234) #2021 90 110
    dataset_train, dataset_ = split_dataset(data_geo_list, 0.8)
    # # dataset_ = random_shuffle(dataset_, 110)
    dataset_dev, dataset_test = split_dataset(dataset_, 0.5)

    dir_input = ('../dataset/' + DATASET + '/input/'
                 'radius' + str(radius) + '_ngram' + str(ngram) + '/final/')
    os.makedirs(dir_input, exist_ok=True)

    # torch.save(dataset_train, dir_input + 'drug-target_train_1000' + ".pt")
    # torch.save(dataset_dev, dir_input + 'drug-target_dev_notcutted_test_1000' + ".pt")
    # torch.save(dataset_test, dir_input + 'drug-target_test_notcutted_test_1000' + ".pt")

    # torch.save(data_geo_list, dir_input + 'drug-target_esmpro_4000' + ".pt")
    torch.save(dataset_train, dir_input + 'drug-target_train_esmpro_4000' + ".pt")
    torch.save(dataset_dev, dir_input + 'drug-target_dev_esmpro_4000' + ".pt")
    torch.save(dataset_test, dir_input + 'drug-target_test_esmpro_4000' + ".pt")

    # with open(dir_input + 'Smiles.txt', 'w') as f:
    #     f.write(Smiles)
    # np.save(dir_input + 'compounds', compounds)
    # np.save(dir_input + 'adjacencies', adjacencies)
    # np.save(dir_input + 'proteins', proteins)
    # np.save(dir_input + 'interactions', interactions)
    # print(len(fingerprint_dict))
    print(len(fingerprint_dict))
    # dictdir_input = ('../dataset/' + 'dict/'
    #              'radius' + str(radius) + '_ngram' + str(ngram)+'/')
    # os.makedirs(dictdir_input, exist_ok=True)
    dump_dictionary(fingerprint_dict, dir_input + 'fingerprint_dict.pickle')
    dump_dictionary(word_dict, dir_input + 'word_dict.pickle')
    dump_dictionary(atom_dict, dir_input + 'atom_dict.pickle')
    dump_dictionary(edge_dict, dir_input + 'edge_dict.pickle')
    dump_dictionary(bond_dict, dir_input + 'bond_dict.pickle')

    print('The preprocess of ' + DATASET + ' dataset has finished!')

