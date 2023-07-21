import torch.nn as nn
import torch
import math
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import numpy as np
from src.args import import_from_string
from .yoso_kernel import kernel as yoso
from ..prenorm import Attention

class YOSOFFN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hashcode_len = config.ffn_hashcode_len
        self.num_hash_f = config.ffn_num_hash_f
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)

        self.k_weight = nn.Parameter(0.02 * torch.randn(self.intermediate_size, self.hidden_size))
        self.q_weight = nn.Parameter(0.02 * torch.randn(self.intermediate_size, self.hidden_size))
        self.bias = nn.Parameter(0.001 * torch.randn(self.hidden_size))

    def _forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        hidden_states = self.LayerNorm(hidden_states)

        Q = hidden_states[None]
        K = self.k_weight[None]
        V = self.q_weight[None]

        Q_mask = torch.ones(batch_size, dtype = torch.int32, device = hidden_states.device)
        K_mask = torch.ones(self.intermediate_size, dtype = torch.int32, device = hidden_states.device)

        if self.num_hash_f < 0:
            Q = yoso.normalize(Q)
            K = yoso.normalize(K)
            X = yoso.yoso_e(Q, K, V, Q_mask, K_mask, self.hashcode_len)
        else:
            if self.training:
                Q = yoso.normalize(Q)
                K = yoso.normalize(K)
            X = yoso.yoso(
                Q.float().contiguous(), K.float().contiguous(), V.float().contiguous(),
                Q_mask.contiguous(), K_mask.contiguous(),
                self.num_hash_f, self.hashcode_len
            ).to(Q.dtype)

        X = yoso.normalize(X) + self.bias

        return X[0]

    def forward(self, hidden_states):
        shape = hidden_states.shape[:-1]
        dim = hidden_states.shape[-1]
        hidden_states = hidden_states.reshape(np.prod(shape).item(), dim)
        outputs = self._forward(hidden_states)
        outputs = outputs.reshape(*shape, self.hidden_size)
        return outputs

    def extra_repr(self):
        return f'num_hash_f={self.num_hash_f}, hashcode_len={self.hashcode_len}, intermediate_size={self.intermediate_size}'

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.FFN = YOSOFFN(config)
        self.attention = Attention(config)
        self.dropout = nn.Dropout(p = config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attn_output = self.dropout(self.attention(hidden_states, attention_mask)) + hidden_states
        layer_output = self.dropout(self.FFN(attn_output)) + attn_output
        return layer_output