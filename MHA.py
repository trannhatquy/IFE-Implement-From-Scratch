# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F
import math


# Single Head Self Attention
class SingleHeadSelfAttention(nn.Module):
    def __init__(self, h_dim):
        super().__init__()

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

    def forward(self, x):
        B, T, D = x.shape  # batch size, sequence length, hidden dimension

        q = self.q_net(x)  # (B, T, D)
        k = self.k_net(x)  # (B, T, D)
        v = self.v_net(x)  # (B, T, D)

        weights = q @ k.transpose(1, 2) / math.sqrt(D)  # (B, T, T)
        normalized_weights = F.softmax(weights, dim=-1)  # (B, T, T)

        attention = normalized_weights @ v  # (B, T, D)

        return attention


# Multi Head Self Attention
class MultiHeadSelfAttention(nn.module):
    def __init__(self, h_dim, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)
        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

    def forward(self, x):
        B, T, C = x.shape  # batch size, sequence length, n_heads * h_dim

        N, D = self.n_heads, C // self.n_heads  # n_heads, D = attention dimension

        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)  # (B, N, T, D)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)  # (B, N, T, D)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)  # (B, N, T, D)

        weights = q @ k.transpose(2, 3) / math.sqrt(D)  # (B, N, T, T)
        normalized_weights = F.softmax(weights, dim=-1)  # (B, N, T, T)

        attention = self.att_drop(normalized_weights @ v)  # (B, N, T, D)

        attention = attention.transpose(
            1, 2).contiguous().view(B, T, N*D)  # (B, T, C)
        out = self.proj_drop(self.proj_net(attention))  # (B, T, C)
        return out
