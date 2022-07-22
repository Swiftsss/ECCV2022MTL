import torch
from torch import nn
from model_code.EmotionNet import *
import utils
import numpy as np
from model_code.BTS_Double_model import BTS_Double_model
from model_code.Single_model import Single_model
from model_code.SMM_model import SMM_coSMM_modelnfig_model
from model_code.TS_Double_model import TS_Double_model
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k) # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn
class MultiHeadAttention(nn.Module):
    def __init__(self, model_hidden, n_heads, hidden_dim):
        super().__init__()
        self.d_q = hidden_dim
        self.d_k = hidden_dim
        self.d_v = hidden_dim
        self.n_heads = n_heads
        self.W_Q = nn.Linear(model_hidden, self.d_k * n_heads)
        self.W_K = nn.Linear(model_hidden, self.d_k * n_heads)
        self.W_V = nn.Linear(model_hidden, self.d_v * n_heads)
        self.linear = nn.Linear(n_heads * self.d_v, model_hidden)
        self.layer_norm = nn.LayerNorm(model_hidden)
    
    def forward(self, Q, K, V):
        residual, batch_size = K, K.size(0)

        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)    # q_s [batch, n_heads, len_q, d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)    # q_s [batch, n_heads, len_k, d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)    # q_s [batch, n_heads, len_k, d_v]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(context)
        return self.layer_norm(output) # output: [batch, len_q, model_hidden]


def get_backbone(args):
    return TS_Double_model(args)