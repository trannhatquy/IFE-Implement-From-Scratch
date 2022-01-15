# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)
        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask', mask)

    def forward(self, x):
        B, T, C = x.shape  # batch size, sequence length, n_heads * h_dim

        N, D = self.n_heads, C // self.n_heads  # n_heads, D = attention dimension

        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)  # (B, N, T, D)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)  # (B, N, T, D)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)  # (B, N, T, D)

        weights = q @ k.transpose(2, 3) / math.sqrt(D)  # (B, N, T, T)
        weights.masked_fill_(self.mask[..., :T, :T] == 0, float(
            '-inf'))  # causal mask applied to weights
        normalized_weights = F.softmax(weights, dim=-1)  # (B, N, T, T)

        attention = self.att_drop(normalized_weights @ v)  # (B, N, T, D)

        attention = attention.transpose(
            1, 2).contiguous().view(B, T, N*D)  # (B, T, C)
        out = self.proj_drop(self.proj_net(attention))  # (B, T, C)
        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)

        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4*h_dim),
            nn.GELU(),
            nn.Linear(4*h_dim, h_dim),
            nn.Dropout(drop_p),
        )

        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attentipon -> LayerNorm -> MLP -> LaynerNorm

        x = x + self.attention(x)  # residual connection
        x = self.ln1(x)
        x = x + self.mlp(x)
        x = self.ln2(x)
        return x


class GPT(nn.Module):
    def __init__(self, token_dim, n_blocks, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        # embed input tokens and positions
        self.proj_token = nn.Embedding(token_dim, h_dim)
        # parameter = trainable weight matrix
        init_param_vals = torch.randn(1, max_T, h_dim) / math.sqrt(h_dim)
        self.position_embedding = nn.Parameter(init_param_vals)
        self.dropout = nn.Dropout(drop_p)

        # transformer blocks
        blocks = [Block(h_dim, max_T, n_heads, drop_p)
                  for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

        # projection head
        self.ln = nn.LayerNorm(h_dim)
        self.proj_head = nn.Linear(h_dim, token_dim)

    def forward(self, x):
        B, T = x.shape

        # token and pos embedding
        x = self.proj_token(x)
        pos_h = self.position_embedding[:, :T, :]
        h = x + pos_h

        # transformers and prediction
        h = self.ln(self.transformer(h))
        pred = self.proj_head(h)

        return pred

    def pred_loss(self, pred, target):
        # pred (B, T, C) and target (B, T)
        B, T, C = pred.shape
        return F.cross_entropy(pred.view(B*T, C), target.view(B*T))
