from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F
import math


def create_kqv_matrix(input_vector_dim, n_heads = 1):
    return nn.Linear(input_vector_dim, 3 * (input_vector_dim // n_heads))

def kqv(x, linear):
    k, q, v = torch.split(linear(x), linear(x).size(-1) // 3, dim=-1)
    return k, q, v

def attention_scores(a, b):
    B1, N1, D1 = a.size()
    B2, N2, D2 = b.size()
    assert B1 == B2
    assert D1 == D2

    return torch.bmm(a, b.transpose(1, 2)) / math.sqrt(D1)

def create_causal_mask(embed_dim, n_heads, max_context_len):
    mask = torch.tril(torch.ones(max_context_len, max_context_len)).view(1, max_context_len, max_context_len)
    return mask

def self_attention(v, A, mask = None):
    if mask is not None:
        A = A.masked_fill(mask == 0, float("-inf"))
    A = F.softmax(A, dim=-1)
    sa = torch.bmm(A, v)
    return sa


def self_attention_layer(x, kqv_matrix, attention_mask):
    k, q, v = kqv(x, kqv_matrix)
    att = attention_scores(k, q)
    sa = self_attention(v, att, attention_mask)
    return sa

def multi_head_attention_layer(x, kqv_matrices, mask):
    B, N, D = x.size()
    outputs = []
    for kqv_matrix in kqv_matrices:
        sa = self_attention_layer(x, kqv_matrix, mask)
        outputs.append(sa)
    sa = torch.cat(outputs, dim=-1)
    assert sa.size() == (B, N, D)
    return sa


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, max_context_len):
        super().__init__()
        assert embed_dim % n_heads == 0
        # the linear layers used for k, q, v computations:
        # each linear is for a different head, but for all of k, q and v for this head.
        self.kqv_matrices = nn.ModuleList([create_kqv_matrix(embed_dim, n_heads) for i in range(n_heads)])
        # For use in the causal part.  "register_buffer" is used to store a tensor which is fixed but is not a parameter of the model.
        # You can then access it with: self.mask
        mask = create_causal_mask(embed_dim, n_heads, max_context_len)
        self.register_buffer("mask", mask)
        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, D = x.size()
        mask = self.mask[:, :N, :N]
        sa = multi_head_attention_layer(x, self.kqv_matrices, mask)
        sa = self.proj(sa)
        return sa
