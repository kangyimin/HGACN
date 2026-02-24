# layers.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np

class hyperedge_encoder(nn.Module):
    def __init__(self, num_in_edge, num_hidden, dropout, act=torch.tanh):
        super(hyperedge_encoder, self).__init__()
        self.num_in_edge = num_in_edge
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act = act
        self.W1 = nn.Parameter(torch.zeros(size=(self.num_in_edge, self.num_hidden), dtype=torch.float))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = nn.Parameter(torch.zeros(self.num_hidden, dtype=torch.float))

    def forward(self, H_T):
        z1 = self.act(torch.mm(H_T, self.W1) + self.b1)
        return z1

class node_encoder(nn.Module):
    def __init__(self, num_in_node, num_hidden, dropout, act=torch.tanh):
        super(node_encoder, self).__init__()
        self.num_in_node = num_in_node
        self.num_hidden = num_hidden
        self.dropout = dropout
        self.act = act
        self.W1 = nn.Parameter(torch.zeros(size=(self.num_in_node, self.num_hidden), dtype=torch.float))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.b1 = nn.Parameter(torch.zeros(self.num_hidden, dtype=torch.float))

    def forward(self, H):
        z1 = self.act(H.mm(self.W1) + self.b1)
        return z1

class decoder2(nn.Module):
    def __init__(self, dropout=0.5):
        super(decoder2, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_node, z_hyperedge):
        z_node_ = self.dropout(z_node)
        z_hyperedge_ = self.dropout(z_hyperedge)
        z = z_node_.mm(z_hyperedge_.t())
        return z

class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x

class HGNN_conv1(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv1, self).__init__()
        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x

class HGNN2(nn.Module):
    def __init__(self, in_ch, n_hid, n_class, n_node, emb_dim, dropout=0.5):
        super(HGNN2, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)
        self.feat = nn.Embedding(n_node, emb_dim)
        self.feat_idx = torch.arange(n_node).long()
        nn.init.xavier_uniform_(self.feat.weight)

    def forward(self, x, G):
        device = x.device
        self.feat_idx = self.feat_idx.to(device)
        x = self.feat(self.feat_idx)
        x = x.squeeze(1)
        x = F.tanh(self.hgc1(x, G))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.hgc2(x, G)
        return x

class HGATConv(nn.Module):
    def __init__(self, in_features, out_features, heads=4, alpha=0.2, dropout=0.3):
        super(HGATConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.head_dim = out_features // heads
        self.alpha = alpha
        self.dropout = dropout

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.zeros(size=(2 * self.head_dim, 1)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, adj):
        h = torch.mm(x, self.W)
        h = h.view(h.size(0), self.heads, self.head_dim)
        h = h.permute(1, 0, 2)

        attn_scores = []
        for head in range(self.heads):
            h_head = h[head]
            a_input = torch.cat([h_head.unsqueeze(1).expand(-1, h_head.size(0), -1),
                               h_head.unsqueeze(0).expand(h_head.size(0), -1, -1)], dim=2)
            e_head = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
            zero_vec = -9e15 * torch.ones_like(e_head)
            attn = torch.where(adj > 0, e_head, zero_vec)
            attn = F.softmax(attn, dim=1)
            attn_scores.append(attn)

        h_prime = torch.stack([torch.matmul(attn_scores[i], h[i]) for i in range(self.heads)])
        h_prime = h_prime.permute(1, 0, 2).contiguous().view(h.size(1), -1)
        return F.elu(h_prime)

class HGAT(nn.Module):
    def __init__(self, in_features, out_features, heads=4, dropout=0.3):
        super(HGAT, self).__init__()
        self.heads = heads
        self.dropout = dropout
        self.conv1 = HGATConv(in_features, out_features, heads=heads, dropout=dropout)
        self.conv2 = HGATConv(out_features, out_features, heads=heads, dropout=dropout)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv1(x, adj)
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, adj)
        return x

class self_Attention(nn.Module):
    def __init__(self, num, num_in, num_hidden):
        super(self_Attention, self).__init__()
        self.num = num
        self.num_in = num_in
        self.hidden = num_hidden
        self.Wr = nn.Parameter(torch.zeros(size=(self.num_in, self.hidden), dtype=torch.float))
        nn.init.xavier_uniform_(self.Wr.data, gain=1.0)
        self.b1 = nn.Parameter(torch.zeros(self.hidden, dtype=torch.float))
        self.P = nn.Parameter(torch.zeros(size=(self.hidden, self.hidden), dtype=torch.float))
        nn.init.xavier_uniform_(self.P.data, gain=1.0)
        self.Mr = nn.Parameter(torch.zeros(size=(self.num, self.num), dtype=torch.float))

    def forward(self, embedding):
        intermediate = embedding.mm(self.Wr) + self.b1
        alpha = F.tanh(intermediate).mm(self.P)
        return alpha