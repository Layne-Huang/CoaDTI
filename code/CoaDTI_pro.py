import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

import torch

from torch_geometric.utils import add_self_loops, degree, remove_self_loops
from torch_geometric.nn import MessagePassing
from transformers import XLNetModel, BertTokenizer, pipeline, BertModel, AlbertModel
import esm
import math

embed_dim = 128
device = torch.device('cuda')
esm_model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()

class MHAtt(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(MHAtt, self).__init__()

        self.linear_v = nn.Linear(hid_dim, hid_dim)
        self.linear_k = nn.Linear(hid_dim, hid_dim)
        self.linear_q = nn.Linear(hid_dim, hid_dim)
        self.linear_merge = nn.Linear(hid_dim, hid_dim)
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.nhead = n_heads

        self.dropout = nn.Dropout(dropout)
        self.hidden_size_head = int(self.hid_dim / self.nhead)
    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.nhead,
            self.hidden_size_head
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.nhead,
            self.hidden_size_head
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.nhead,
            self.hidden_size_head
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hid_dim
        )


        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)

class DPA(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(DPA, self).__init__()

        self.mhatt1 = MHAtt(hid_dim, n_heads, dropout)
        # self.mhatt1 = MultiAttn(hid_dim, n_heads)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hid_dim)



    def forward(self, x, y, y_mask=None):

        # x as V while y as Q and K
        # x = self.norm1(x + self.dropout1(
        #     self.mhatt1(x, x, y, y_mask)
        # ))
        x = self.norm1(x+self.dropout1(
            self.mhatt1(y, y, x, y_mask)
        ))
        # x = self.norm1(x + self.dropout1(
        #     self.mhatt1(x, y, y_mask, y_mask)
        # ))

        return x

class SA(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super(SA, self).__init__()

        self.mhatt1 = MHAtt(hid_dim, n_heads, dropout)
        # self.mhatt1 = MultiAttn(hid_dim, n_heads)

        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hid_dim)



    def forward(self, x, mask=None):

        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, mask)
        ))
        # x = self.norm1(x + self.dropout1(
        #     self.mhatt1(x, x, mask, mask)
        # ))

        return x

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, out_channels]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr='max')  # "Max" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(in_channels + out_channels, in_channels, bias=False)
        self.update_act = torch.nn.ReLU()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        a = edge_index
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        # x_j has shape [E, in_channels]

        x_j = self.lin(x_j)
        x_j = self.act(x_j)

        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]

        new_embedding = torch.cat([aggr_out, x], dim=1)

        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)

        return new_embedding

class gnn(nn.Module):
    def __init__(self, n_fingerprint, pooling):
        super(gnn, self).__init__()
        self.pooling = pooling
        self.embed_fingerprint = nn.Embedding(num_embeddings=n_fingerprint, embedding_dim=embed_dim)
        self.conv1 = SAGEConv(embed_dim, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = SAGEConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = SAGEConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.linp1 = torch.nn.Linear(256, 128)
        self.linp2 = torch.nn.Linear(128, 512)

        # self.lin1 = torch.nn.Linear(128, 256)
        # self.lin2 = torch.nn.Linear(256, 512)
        # self.lin3 = torch.nn.Linear(64, 90)
        self.lin = torch.nn.Linear(128, 512)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def forward(self, data):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        x, edge_index ,batch= data.x, data.edge_index,data.batch
        x = self.embed_fingerprint(x)

        x = x.squeeze(1)
        # print("after conv1:\t", self.conv1(x, edge_index).shape)# print(type(x))
        x = F.relu(self.conv1(x, edge_index))

        if self.pooling:
            x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
            x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        # print(x.shape)
        if self.pooling:
            x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
            x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # x = F.relu(self.conv3(x, edge_index))

        if self.pooling:
            x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
            x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

            x = x1 + x2 + x3
            x = self.linp1(x)
            x = self.act1(x)
            x = self.linp2(x)
        if not self.pooling:
            # x = self.lin1(x)
            # x = self.act1(x)
            # x = self.lin2(x)
            x = self.lin(x)
        # x = self.lin(x)
        # x = self.act2(x)
        # x = self.lin3(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        return x

class inter_cross_att(nn.Module):
    def __init__(self, dim, nhead, dropout):
        super(inter_cross_att, self).__init__()
        self.sca = SA(dim, nhead, dropout)
        self.spa = SA(dim, nhead, dropout)
        self.coa_pc = DPA(dim, nhead, dropout)
        self.coa_cp = DPA(dim, nhead, dropout)

    def forward(self, protein_vector, compound_vector):
        compound_vector = self.sca(compound_vector, None)  # self-attention
        protein_vector = self.spa(protein_vector, None)  # self-attention
        compound_covector = self.coa_pc(compound_vector, protein_vector, None)  # co-attention
        protein_covector = self.coa_cp(protein_vector, compound_vector, None)  # co-attention

        return protein_covector, compound_covector


class encoder_cross_att_old(nn.Module):
    def __init__(self, dim, nhead, dropout):
        super(encoder_cross_att_old, self).__init__()
        self.sca = SA(dim, nhead, dropout)
        self.spa = SA(dim, nhead, dropout)
        self.coa_cp = DPA(dim, nhead, dropout)

    def forward(self, protein_vector, compound_vector):
        compound_vector = self.sca(compound_vector, None)  # self-attention
        protein_vector = self.spa(protein_vector, None)  # self-attention
        protein_covector = self.coa_cp(protein_vector, compound_vector, None)  # co-attention

        return protein_covector, compound_vector

class stack_cross_att(nn.Module):
    def __init__(self, dim, nhead, dropout):
        super(stack_cross_att, self).__init__()
        self.sca = SA(dim, nhead, dropout)
        self.spa = SA(dim, nhead, dropout)
        self.coa_cp = DPA(dim, nhead, dropout)

    def forward(self, protein_vector, compound_vector):
        compound_vector = self.sca(compound_vector, None)  # self-attention
        protein_vector = self.spa(protein_vector, None)  # self-attention
        protein_covector = self.coa_cp(protein_vector, compound_vector, None)  # co-attention

        return protein_covector, compound_vector

class encoder_cross_att(nn.Module):
    def __init__(self, dim, nhead, dropout, layers):
        super(encoder_cross_att, self).__init__()
        # self.encoder_layers = nn.ModuleList([SEA(dim, dropout) for _ in range(layers)])
        self.encoder_layers = nn.ModuleList([SA(dim, nhead, dropout) for _ in range(layers)])
        self.decoder_sa = nn.ModuleList([SA(dim, nhead, dropout) for _ in range(layers)])
        self.decoder_coa = nn.ModuleList([DPA(dim, nhead, dropout)  for _ in range(layers)])
        self.layer_coa = layers
    def forward(self, protein_vector, compound_vector):
        for i in range(self.layer_coa):
            compound_vector = self.encoder_layers[i](compound_vector, None)  # self-attention
        for i in range(self.layer_coa):
            protein_vector = self.decoder_sa[i](protein_vector, None)
            protein_vector = self.decoder_coa[i](protein_vector, compound_vector, None)# co-attention

        return protein_vector, compound_vector



class Dtis_old(nn.Module):
    def __init__(self, n_fingerprint, dim, n_word, window, layer_output, layer_coa, nhead=8, dropout=0.1, encoder=False):
        super(Dtis_old, self).__init__()
        self.encoder = encoder
        self.layer_output = layer_output
        self.layer_coa = layer_coa
        # self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        # self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
        #                             for _ in range(layer_gnn)])
        self.gnn = gnn(n_fingerprint, encoder)
        self.esmmodel = esm_model
        # self.protBert = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
        # self.protBert = AlbertModel.from_pretrained("Rostlab/prot_albert")

        self.W_attention = nn.Linear(dim, dim)

        self.sca_1 = SA(dim, nhead, dropout)
        self.sca_2 = SA(dim, nhead, dropout)
        self.sca_3 = SA(dim, nhead, dropout)

        # self-protein-attention layers
        self.spa_1 = SA(dim, nhead, dropout)
        self.spa_2 = SA(dim, nhead, dropout)
        self.spa_3 = SA(dim, nhead, dropout)


        self.coa_pc_1 = DPA(dim, nhead, dropout)
        self.coa_pc_2 = DPA(dim, nhead, dropout)
        self.coa_cp_1 = DPA(dim, nhead, dropout)
        self.coa_cp_2 = DPA(dim, nhead, dropout)
        self.coa_pc_3 = DPA(dim, nhead, dropout)
        self.coa_cp_3 = DPA(dim, nhead, dropout)

        self.W_out = nn.ModuleList([nn.Linear(2 * dim, dim),nn.Linear(dim, 128),nn.Linear(128, 64)
                                    ])

        self.inter_coa_layers = nn.ModuleList([inter_cross_att(dim, nhead, dropout) for _ in range(layer_coa)])
        self.encoder_coa_layers = nn.ModuleList([encoder_cross_att_old(dim, nhead, dropout) for _ in range(layer_coa)])

        self.lin = nn.Linear(768, 512) #bert1024 esm768
        # self.lin = nn.Linear(4096, 512)
        self.W_interaction = nn.Linear(64, 2)



    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""

        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs


        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs, proteins):

        fingerprints, adjacency, words = inputs.x, inputs.edge_index, inputs.protein
        # print(words)
        # print(type(words[0]))
        # exit()
        # print("x.shape:", fingerprints.shape)
        # print(fingerprints.shape)
        """Compound vector with GNN."""
        # fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(inputs)
        compound_vector = torch.unsqueeze(compound_vector, 0)  # sequence-like GNN ouput
        # compound_vector = torch.mean(compound_vector,dim=0,keepdim=True)
        # words = words.to(device)


        """Protein vector with attention-CNN."""
        # word_vectors = self.embed_word(words)

        # protein_vector = self.protBert(input_ids=words['input_ids'], token_type_ids=words['token_type_ids'], attention_mask=words['attention_mask'])[0]
        # for protine in
        # with torch.no_grad():
        #     protein_vector = self.protBert(**proteins)[0]

        with torch.no_grad():
            results = self.esmmodel(proteins, repr_layers=[6])
        token_representations = results["representations"][6]
        # temp = results["logits"]
        protein_vector = token_representations[:, 1:, :]
        protein_vector = self.lin(torch.squeeze(protein_vector, 1))

        # protein_vector = self.MHAtt(protein_vector, protein_vector, compound_vector, None)
        # compound_vector = self.MHAtt(compound_vector, compound_vector, compound_vector, None)
        # protein_vector = self.MHAtt(protein_vector, protein_vector, compound_vector, None)
        # protein_vector = self.MHAtt(protein_vector, protein_vector, compound_vector, None)
        # compound_vector = self.MHAtt(compound_vector, compound_vector, compound_vector, None)
        # protein_vector = self.MHAtt(protein_vector, protein_vector, compound_vector, None)
        # protein_vector = protein_vector.squeeze(1)
        # compound_vector = compound_vector.squeeze(1)

        # intersecting-mode
        # compound_vector = torch.unsqueeze(compound_vector, 1)
        # compound_vector = self.sca_1(compound_vector, None)  # self-attention
        # protein_vector = self.spa_1(protein_vector, None)  # self-attention
        # compound_vector = self.coa_pc_1(compound_vector, protein_vector, None)  # co-attention
        # protein_vector = self.coa_cp_1(protein_vector, compound_vector, None)  # co-attention

        for i in range(self.layer_coa):
            # protein_vector, compound_vector = self.inter_coa_layers[i](protein_vector, compound_vector)
            if self.encoder:
                protein_vector, compound_vector = self.encoder_coa_layers[i](protein_vector, compound_vector)
            else:
                protein_vector, compound_vector = self.inter_coa_layers[i](protein_vector, compound_vector)
        protein_vector = protein_vector.mean(dim=1)
        compound_vector = compound_vector.mean(dim=1)
        # compound_vector = compound_vector.squeeze(1) #batch
        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(self.layer_output):
            cat_vector = torch.tanh(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)
        # print(interaction)
        # print(F.softmax(interaction, 1).to('cpu').data.numpy())
        # exit()
        return interaction

    def __call__(self, data, proteins, train=True):

        # inputs = data.x, data.edge_index, data.protein
        correct_interaction = data.y

        predicted_interaction = self.forward(data, proteins)

        if train:
            criterion = torch.nn.CrossEntropyLoss().to(device)
            loss = criterion(predicted_interaction, correct_interaction)
            return loss, predicted_interaction
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores

class Dtis(nn.Module):
    def __init__(self, n_fingerprint, dim, n_word, window, layer_output, layer_coa, nhead=8, dropout=0.1, co_attention=False, gcn_pooling =False):
        super(Dtis, self).__init__()
        self.co_attention = co_attention
        self.layer_output = layer_output
        self.layer_coa = layer_coa
        # self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        # self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
        #                             for _ in range(layer_gnn)])
        self.gnn = gnn(n_fingerprint, gcn_pooling)
        self.esmmodel = esm_model
        # self.protBert = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
        # self.protBert = AlbertModel.from_pretrained("Rostlab/prot_albert")

        self.W_attention = nn.Linear(dim, dim)

        self.sca_1 = SA(dim, nhead, dropout)
        self.sca_2 = SA(dim, nhead, dropout)
        self.sca_3 = SA(dim, nhead, dropout)

        # self-protein-attention layers
        self.spa_1 = SA(dim, nhead, dropout)
        self.spa_2 = SA(dim, nhead, dropout)
        self.spa_3 = SA(dim, nhead, dropout)


        self.coa_pc_1 = DPA(dim, nhead, dropout)
        self.coa_pc_2 = DPA(dim, nhead, dropout)
        self.coa_cp_1 = DPA(dim, nhead, dropout)
        self.coa_cp_2 = DPA(dim, nhead, dropout)
        self.coa_pc_3 = DPA(dim, nhead, dropout)
        self.coa_cp_3 = DPA(dim, nhead, dropout)

        self.W_out = nn.ModuleList([nn.Linear(2 * dim, dim),nn.Linear(dim, 128),nn.Linear(128, 64)
                                    ])

        self.encoder_coa_layers = encoder_cross_att(dim, nhead, dropout, layer_coa)
        self.inter_coa_layers = nn.ModuleList([inter_cross_att(dim, nhead, dropout) for _ in range(layer_coa)])
        self.stack_coa_layers = nn.ModuleList([stack_cross_att(dim, nhead, dropout) for _ in range(layer_coa)])

        self.lin = nn.Linear(768, 512) #bert1024 esm768
        # self.lin = nn.Linear(4096, 512)
        self.W_interaction = nn.Linear(64, 2)



    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""

        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = torch.relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs


        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs, proteins):

        fingerprints, adjacency, words = inputs.x, inputs.edge_index, inputs.protein
        # print(words)
        # print(type(words[0]))
        # exit()
        # print("x.shape:", fingerprints.shape)
        # print(fingerprints.shape)
        """Compound vector with GNN."""
        # fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(inputs)
        compound_vector = torch.unsqueeze(compound_vector, 0)  # sequence-like GNN ouput
        # compound_vector = torch.mean(compound_vector,dim=0,keepdim=True)
        # words = words.to(device)


        """Protein vector with attention-CNN."""
        # word_vectors = self.embed_word(words)

        # protein_vector = self.protBert(input_ids=words['input_ids'], token_type_ids=words['token_type_ids'], attention_mask=words['attention_mask'])[0]
        # for protine in
        # with torch.no_grad():
        #     protein_vector = self.protBert(**proteins)[0]

        with torch.no_grad():
            results = self.esmmodel(proteins, repr_layers=[6])
        token_representations = results["representations"][6]
        # temp = results["logits"]
        protein_vector = token_representations[:, 1:, :]
        protein_vector = self.lin(torch.squeeze(protein_vector, 1))

        # protein_vector = self.MHAtt(protein_vector, protein_vector, compound_vector, None)
        # compound_vector = self.MHAtt(compound_vector, compound_vector, compound_vector, None)
        # protein_vector = self.MHAtt(protein_vector, protein_vector, compound_vector, None)
        # protein_vector = self.MHAtt(protein_vector, protein_vector, compound_vector, None)
        # compound_vector = self.MHAtt(compound_vector, compound_vector, compound_vector, None)
        # protein_vector = self.MHAtt(protein_vector, protein_vector, compound_vector, None)
        # protein_vector = protein_vector.squeeze(1)
        # compound_vector = compound_vector.squeeze(1)

        # intersecting-mode
        # compound_vector = torch.unsqueeze(compound_vector, 1)
        # compound_vector = self.sca_1(compound_vector, None)  # self-attention
        # protein_vector = self.spa_1(protein_vector, None)  # self-attention
        # compound_vector = self.coa_pc_1(compound_vector, protein_vector, None)  # co-attention
        # protein_vector = self.coa_cp_1(protein_vector, compound_vector, None)  # co-attention

        if 'encoder' in self.co_attention:
            protein_vector, compound_vector = self.encoder_coa_layers(protein_vector, compound_vector)
        elif 'stack' in self.co_attention:
            for i in range(self.layer_coa):
                protein_vector, compound_vector = self.stack_coa_layers[i](protein_vector, compound_vector)
        else:
            for i in range(self.layer_coa):
                protein_vector, compound_vector = self.inter_coa_layers[i](protein_vector, compound_vector)
        protein_vector = protein_vector.mean(dim=1)
        compound_vector = compound_vector.mean(dim=1)
        # compound_vector = compound_vector.squeeze(1) #batch
        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(self.layer_output):
            cat_vector = torch.tanh(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)
        # print(interaction)
        # print(F.softmax(interaction, 1).to('cpu').data.numpy())
        # exit()
        return interaction

    def __call__(self, data, proteins, train=True):

        # inputs = data.x, data.edge_index, data.protein
        correct_interaction = data.y

        predicted_interaction = self.forward(data, proteins)

        if train:
            criterion = torch.nn.CrossEntropyLoss().to(device)
            loss = criterion(predicted_interaction, correct_interaction)
            return loss, predicted_interaction
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores

class Dtis_ablation(nn.Module):
    def __init__(self, n_fingerprint, dim, n_word, layer_output, encoder=False):
        super(Dtis_ablation, self).__init__()
        self.encoder = encoder
        self.layer_output = layer_output

        self.embed_word = nn.Embedding(n_word, dim)
        self.gnn = gnn(n_fingerprint, encoder)
        self.esmmodel = esm_model

        self.W_out = nn.ModuleList([nn.Linear(2 * dim, dim),nn.Linear(dim, 128),nn.Linear(128, 64)
                                    ])

        self.lin = nn.Linear(768, 512) #bert1024 esm768

        self.W_interaction = nn.Linear(64, 2)


    def forward(self, inputs, proteins):

        """Compound vector with GNN."""
        compound_vector = self.gnn(inputs)
        compound_vector = torch.unsqueeze(compound_vector, 0)  # sequence-like GNN ouput



        """Protein vector with attention-CNN."""

        with torch.no_grad():
            results = self.esmmodel(proteins, repr_layers=[6])
        token_representations = results["representations"][6]
        protein_vector = token_representations[:, 1:, :]
        # protein_vector = self.lin(torch.squeeze(protein_vector, 1))
        protein_vector = self.lin(protein_vector)

        protein_vector = protein_vector.mean(dim=1)
        compound_vector = compound_vector.mean(dim=1)

        """Concatenate the above two vectors and output the interaction."""
        try:
            cat_vector = torch.cat((compound_vector, protein_vector), 1)
        except:
            print(compound_vector.size())
            print(protein_vector.size())
            exit()
        for j in range(self.layer_output):
            cat_vector = torch.tanh(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

        return interaction

    def __call__(self, data, proteins, train=True):

        correct_interaction = data.y

        predicted_interaction = self.forward(data, proteins)

        if train:
            criterion = torch.nn.CrossEntropyLoss().to(device)
            loss = criterion(predicted_interaction, correct_interaction)
            return loss, predicted_interaction
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores