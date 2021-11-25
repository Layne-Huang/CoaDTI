import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import add_self_loops, degree, remove_self_loops
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch


from transformers import XLNetModel, BertTokenizer, pipeline, BertModel

import math

embed_dim = 128
device = torch.device('cuda')

def gelu(x):
    out = 1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
    return out * x / 2

class PositionalEncoding(nn.Module):
    def __init__(self, d_model,dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class LearnedPositionEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1,max_len = 5000):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):
        weight = self.weight.data.unsqueeze(1)
        x = x + weight[:x.size(0), :]
        return self.dropout(x)

class Att(nn.Module):
    def __init__(self, hid_dim, dropout):
        super(Att, self).__init__()

        self.linear_v = nn.Linear(hid_dim, hid_dim)
        self.linear_k = nn.Linear(hid_dim, hid_dim)
        self.linear_q = nn.Linear(hid_dim, hid_dim)
        self.linear_merge = nn.Linear(hid_dim, hid_dim)
        self.hid_dim = hid_dim
        self.dropout = dropout

        self.dropout = nn.Dropout(dropout)

    def forward(self, v, k, q, mask):
        atted = self.att(v, k, q, mask).transpose(-1,-2)
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

# class MultiAttn(nn.Module):
#     def __init__(self, in_dim, head_num=8):
#         super(MultiAttn, self).__init__()
#
#         self.head_dim = in_dim // head_num
#         self.head_num = head_num
#
#         # scaled dot product attention
#         self.scale = self.head_dim ** -0.5
#
#         self.w_qs = nn.Linear(in_dim, head_num * self.head_dim, bias=True)
#         self.w_ks = nn.Linear(in_dim, head_num * self.head_dim, bias=True)
#         self.w_vs = nn.Linear(in_dim, head_num * self.head_dim, bias=True)
#
#         self.w_os = nn.Linear(head_num * self.head_dim, in_dim, bias=True)
#
#         self.gamma = nn.Parameter(torch.FloatTensor([0]))
#
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x, y, attn_mask, non_pad_mask):
#         B, L, H = x.size()
#         head_num = self.head_num
#         head_dim = self.head_dim
#
#         q = self.w_qs(y).view(B * head_num, L, head_dim)
#         k = self.w_ks(y).view(B * head_num, L, head_dim)
#         v = self.w_vs(x).view(B * head_num, L, head_dim)
#
#         if attn_mask is not None:
#             attn_mask = attn_mask.repeat(head_num, 1, 1)
#
#         attn = torch.bmm(q, k.transpose(1, 2))  # B*head_num, L, L
#         attn = self.scale * attn
#         if attn_mask is not None:
#             attn_mask = attn_mask.repeat(head_num, 1, 1)
#             attn = attn.masked_fill_(attn_mask, -np.inf)
#         attn = self.softmax(attn)
#
#         out = torch.bmm(attn, v)  # B*head_num, L, head_dim
#
#         out = out.view(B, L, head_dim * head_num)
#
#         out = self.w_os(out)
#
#         if non_pad_mask is not None:
#             attn_mask = attn_mask.repeat(head_num, 1, 1)
#             out = non_pad_mask * out
#
#         out = self.gamma * out + x
#
#         return out

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

class SEA(nn.Module):
    def __init__(self, hid_dim, dropout):
        super(SEA, self).__init__()

        self.mhatt1 = Att(hid_dim, dropout)
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
        x, edge_index, batch = data.x, data.edge_index, data.batch
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


class Dtis(nn.Module):
    def __init__(self, n_fingerprint, dim, n_word, layer_output, layer_coa,
                 d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, co_attention=False, gcn_pooling =False):
        super(Dtis, self).__init__()

        self.co_attention = co_attention
        # embedding layer
        self.embed_word = nn.Embedding(n_word, dim) # we do not use additional embedding if we use w2v
        self.pos_encoder = PositionalEncoding(d_model=512) #if w2v d_model=100 else:512
        # self.pos_encoder = LearnedPositionEncoding(d_model=512)
        # feature extraction layer

        self.gnn = gnn(n_fingerprint, gcn_pooling)
        prot_encoder_layer = nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward,dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.prot_encoder = nn.TransformerEncoder(prot_encoder_layer, num_encoder_layers, encoder_norm)
        # self.fc = nn.Linear(100, d_model)

        # attention layers
        self.layer_coa = layer_coa
        # self.encoder_coa_layers = nn.ModuleList([encoder_cross_att(dim, nhead, dropout) for _ in range(layer_coa)])
        self.encoder_coa_layers = encoder_cross_att(dim, nhead, dropout, layer_coa)
        self.inter_coa_layers = nn.ModuleList([inter_cross_att(dim, nhead, dropout) for _ in range(layer_coa)])
        self.stack_coa_layers = nn.ModuleList([stack_cross_att(dim, nhead, dropout) for _ in range(layer_coa)])


        # output layers
        self.layer_output = layer_output
        # self.W_out = nn.ModuleList([nn.Linear(2 * dim, 2 * dim)
        #                             for _ in range(layer_output)])

        self.W_out = nn.ModuleList([nn.Linear(2 * dim, dim),nn.Linear(dim, 128),nn.Linear(128, 64)
                                    ])

        self.W_interaction = nn.Linear(64, 2)

        # self._init_weight()


    def forward(self, inputs, proteins):

        """Compound vector with GNN."""
        compound_vector = self.gnn(inputs)
        # compound_vector = torch.unsqueeze(compound_vector, 0) #no-radius

        """Protein vector with attention-CNN."""
        # proteins = torch.unsqueeze(proteins, 0)
        protein_vector = self.embed_word(proteins)
        # protein_vector = torch.unsqueeze(protein_vector, 0)
        protein_vector = self.pos_encoder(protein_vector)
        # protein_vector = self.fc(protein_vector) #w2v
        protein_vector = self.prot_encoder(protein_vector)
        # protein_vector = self.fc(protein_vector)


        compound_vector = compound_vector.unsqueeze(0)
        if 'encoder' in self.co_attention:
            protein_vector, compound_vector = self.encoder_coa_layers(protein_vector, compound_vector)
        elif 'stack' in self.co_attention:
            for i in range(self.layer_coa):
                protein_vector, compound_vector = self.stack_coa_layers[i](protein_vector, compound_vector)
        else:
            for i in range(self.layer_coa):
                protein_vector, compound_vector = self.inter_coa_layers[i](protein_vector, compound_vector)


        protein_vector = protein_vector.mean(dim=1)
        # compound_vector = compound_vector.squeeze(1) #batch
        compound_vector = compound_vector.mean(dim=1)
        """Concatenate the above two vectors and output the interaction."""
        # catenate the two vectors
        cat_vector = torch.cat((compound_vector, protein_vector), 1)

        # sumarise the two vectors
        # cat_vector = compound_vector+protein_vector
        for j in range(self.layer_output):
            cat_vector = torch.tanh(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)

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
    def __init__(self, n_fingerprint, dim, n_word, layer_output, layer_coa,
                 d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, gcn_pooling =False):
        super(Dtis_ablation, self).__init__()

        # embedding layer
        self.embed_word = nn.Embedding(n_word, dim) # we do not use additional embedding if we use w2v
        self.pos_encoder = PositionalEncoding(d_model=512) #if w2v d_model=100 else:512
        # self.pos_encoder = LearnedPositionEncoding(d_model=512)
        # feature extraction layer

        self.gnn = gnn(n_fingerprint, gcn_pooling)
        prot_encoder_layer = nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward,dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.prot_encoder = nn.TransformerEncoder(prot_encoder_layer, num_encoder_layers, encoder_norm)
        # self.fc = nn.Linear(100, d_model)

        # attention layers
        self.layer_coa = layer_coa
        # self.encoder_coa_layers = encoder_cross_att(dim, nhead, dropout, layer_coa)
        # self.inter_coa_layers = nn.ModuleList([inter_cross_att(dim, nhead, dropout) for _ in range(layer_coa)])
        # self.stack_coa_layers = nn.ModuleList([stack_cross_att(dim, nhead, dropout) for _ in range(layer_coa)])


        # output layers
        self.layer_output = layer_output
        # self.W_out = nn.ModuleList([nn.Linear(2 * dim, 2 * dim)
        #                             for _ in range(layer_output)])

        self.W_out = nn.ModuleList([nn.Linear(2 * dim, dim),nn.Linear(dim, 128),nn.Linear(128, 64)
                                    ])

        self.W_interaction = nn.Linear(64, 2)

        # self._init_weight()


    def forward(self, inputs, proteins):

        """Compound vector with GNN."""
        compound_vector = self.gnn(inputs)
        # compound_vector = torch.unsqueeze(compound_vector, 0) #no-radius

        """Protein vector with attention-CNN."""
        # proteins = torch.unsqueeze(proteins, 0)
        protein_vector = self.embed_word(proteins)
        # protein_vector = torch.unsqueeze(protein_vector, 0)
        protein_vector = self.pos_encoder(protein_vector)
        # protein_vector = self.fc(protein_vector) #w2v
        protein_vector = self.prot_encoder(protein_vector)
        # protein_vector = self.fc(protein_vector)


        compound_vector = compound_vector.unsqueeze(0)



        protein_vector = protein_vector.mean(dim=1)
        compound_vector = compound_vector.mean(dim=1)
        """Concatenate the above two vectors and output the interaction."""
        # catenate the two vectors
        cat_vector = torch.cat((compound_vector, protein_vector), 1)

        # sumarise the two vectors
        # cat_vector = compound_vector+protein_vector
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
            # correct_interaction = torch.tensor(correct_interaction, dtype=torch.long)
            loss = criterion(predicted_interaction, correct_interaction)
            return loss, predicted_interaction
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores

class Dtis_ablation_trans(nn.Module):
    def __init__(self, n_fingerprint, dim, n_word, layer_output, layer_coa,
                 d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1, co_attention=False, gcn_pooling=False):
        super(Dtis_ablation_trans, self).__init__()

        self.co_attention = co_attention
        # embedding layer
        self.embed_word = nn.Embedding(n_word, dim) # we do not use additional embedding if we use w2v
        self.pos_encoder = PositionalEncoding(d_model=512) #if w2v d_model=100 else:512
        # self.pos_encoder = LearnedPositionEncoding(d_model=512)
        # feature extraction layer

        self.gnn = gnn(n_fingerprint, gcn_pooling)
        prot_encoder_layer = nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward,dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.protein_encoder = nn.TransformerEncoder(prot_encoder_layer, num_encoder_layers, encoder_norm)
        # self.fc = nn.Linear(100, d_model)

        # attention layers
        self.layer_coa = layer_coa
        # self.encoder_coa_layers = encoder_cross_att(dim, nhead, dropout, layer_coa)
        self.inter_coa_layers = nn.ModuleList([inter_cross_att(dim, nhead, dropout) for _ in range(layer_coa)])
        # self.stack_coa_layers = nn.ModuleList([stack_cross_att(dim, nhead, dropout) for _ in range(layer_coa)])


        # output layers
        self.layer_output = layer_output
        # self.W_out = nn.ModuleList([nn.Linear(2 * dim, 2 * dim)
        #                             for _ in range(layer_output)])

        self.W_out = nn.ModuleList([nn.Linear(2 * dim, dim),nn.Linear(dim, 128),nn.Linear(128, 64)
                                    ])

        self.W_interaction = nn.Linear(64, 2)

        # self._init_weight()


    def forward(self, inputs, proteins):

        """Compound vector with GNN."""
        compound_vector = self.gnn(inputs)
        # compound_vector = torch.unsqueeze(compound_vector, 0) #no-radius

        """Protein vector with attention-CNN."""
        # proteins = torch.unsqueeze(proteins, 0)
        protein_vector = self.embed_word(proteins)
        # protein_vector = torch.unsqueeze(protein_vector, 0)
        # protein_vector = self.pos_encoder(protein_vector)
        # protein_vector = self.fc(protein_vector) #w2v
        # protein_vector = self.encoder(protein_vector)
        # protein_vector = self.fc(protein_vector)


        compound_vector = compound_vector.unsqueeze(0)

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
        """Concatenate the above two vectors and output the interaction."""
        # catenate the two vectors
        cat_vector = torch.cat((compound_vector, protein_vector), 1)

        # sumarise the two vectors
        # cat_vector = compound_vector+protein_vector
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
            # correct_interaction = torch.tensor(correct_interaction, dtype=torch.long)
            loss = criterion(predicted_interaction, correct_interaction)
            return loss, predicted_interaction
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores

class Dtis_ablation_gnn(nn.Module):
    def __init__(self, n_fingerprint, dim, n_word, layer_output, layer_coa,
                 d_model=512, nhead=8, num_encoder_layers=3, dim_feedforward=1024, dropout=0.1, co_attention=False):
        super(Dtis_ablation_gnn, self).__init__()

        self.co_attention = co_attention
        # embedding layer
        self.embed_word = nn.Embedding(n_word, dim) # we do not use additional embedding if we use w2v
        self.embed_fingerprint = nn.Embedding(num_embeddings=n_fingerprint, embedding_dim=dim)
        self.pos_encoder = PositionalEncoding(d_model=512) #if w2v d_model=100 else:512


        prot_encoder_layer = nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward,dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.protein_encoder = nn.TransformerEncoder(prot_encoder_layer, num_encoder_layers, encoder_norm)

        # attention layers
        self.layer_coa = layer_coa
        # self.encoder_coa_layers = encoder_cross_att(dim, nhead, dropout, layer_coa)
        self.inter_coa_layers = nn.ModuleList([inter_cross_att(dim, nhead, dropout) for _ in range(layer_coa)])
        # self.stack_coa_layers = nn.ModuleList([stack_cross_att(dim, nhead, dropout) for _ in range(layer_coa)])


        # output layers
        self.layer_output = layer_output
        self.W_out = nn.ModuleList([nn.Linear(2 * dim, dim),nn.Linear(dim, 128),nn.Linear(128, 64)
                                    ])

        self.W_interaction = nn.Linear(64, 2)



    def forward(self, inputs, proteins):

        """Compound vector with GNN."""
        compound_vector = self.embed_fingerprint(inputs.x)
        # compound_vector = torch.unsqueeze(compound_vector, 0) #no-radius

        """Protein vector with attention-CNN."""
        # proteins = torch.unsqueeze(proteins, 0)
        protein_vector = self.embed_word(proteins)
        protein_vector = self.pos_encoder(protein_vector)
        protein_vector = self.protein_encoder(protein_vector)


        compound_vector = compound_vector.unsqueeze(0)
        compound_vector = compound_vector.squeeze(2)

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
        """Concatenate the above two vectors and output the interaction."""
        # catenate the two vectors
        cat_vector = torch.cat((compound_vector, protein_vector), 1)

        # sumarise the two vectors
        # cat_vector = compound_vector+protein_vector
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
            # correct_interaction = torch.tensor(correct_interaction, dtype=torch.long)
            loss = criterion(predicted_interaction, correct_interaction)
            return loss, predicted_interaction
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores