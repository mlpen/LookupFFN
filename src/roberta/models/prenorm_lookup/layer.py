import torch.nn as nn
import torch
import math
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import numpy as np
from .ops import query_tables, construct_tables, mra_diffuse
from .fast_hadamard_transform.autograd import hadamard as ht
from src.args import import_from_string
from .yoso_kernel import kernel as yoso

class SBDHT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        if int(2 ** math.ceil(math.log2(self.hidden_size))) != self.hidden_size:
            self.hidden_size = int(2 ** math.ceil(math.log2(self.hidden_size)))

        self.normalized_ht = False
        if hasattr(config, "normalized_ht"):
            self.normalized_ht = config.normalized_ht

        self.block_size = config.block_size
        if self.normalized_ht:
            self.coeff = math.sqrt(self.block_size)
        else:
            self.coeff = math.sqrt(self.block_size * self.hidden_size)

        self.num_ht_iters = config.num_ht_iters
        self.ht_decay_coeff = config.ht_decay_coeff
        self.num_block = self.hidden_size // self.block_size
        self.weight = nn.Parameter(torch.randn(self.num_ht_iters, 2, self.num_block, self.block_size, self.block_size) / self.coeff)
        self.bias = nn.Parameter(0.001 * torch.randn(self.hidden_size))

    def extra_repr(self):
        strs = [f"normalized_ht={self.normalized_ht}"]
        strs.append(f"adjusted_hidden_size={self.hidden_size}, num_ht_iters={self.num_ht_iters}, block_size={self.block_size}, ht_decay_coeff={self.ht_decay_coeff}")
        return "\n".join(strs)

    def _forward(self, hidden_state):
        batch_size, dim = hidden_state.shape
        if dim < self.hidden_size:
            padding_dim = self.hidden_size - dim
            hidden_state = torch.cat([hidden_state, torch.zeros(batch_size, padding_dim, dtype = hidden_state.dtype, device = hidden_state.device)], dim = -1).contiguous()

        inp = hidden_state
        for i in range(self.num_ht_iters):

            hidden_state = hidden_state.reshape(batch_size, self.num_block, self.block_size)
            hidden_state = torch.einsum("bni,noi->bno", hidden_state, self.weight[i, 0])
            hidden_state = hidden_state.reshape(batch_size, self.num_block * self.block_size)

            hidden_state = ht(hidden_state)
            if self.normalized_ht:
                hidden_state = hidden_state / math.sqrt(self.hidden_size)

            hidden_state = hidden_state.reshape(batch_size, self.num_block, self.block_size)
            hidden_state = torch.einsum("bni,noi->bno", hidden_state, self.weight[i, 1])
            hidden_state = hidden_state.reshape(batch_size, self.num_block * self.block_size)

            hidden_state = ht(hidden_state)
            if self.normalized_ht:
                hidden_state = hidden_state / math.sqrt(self.hidden_size)

            hidden_state = hidden_state * self.ht_decay_coeff + inp * (1 - self.ht_decay_coeff)

        hidden_state = hidden_state + self.bias
        if dim < self.hidden_size:
            hidden_state = hidden_state[:, :dim].contiguous()

        return hidden_state

    def forward(self, hidden_states):
        shape = hidden_states.shape[:-1]
        dim = hidden_states.shape[-1]
        hidden_states = hidden_states.reshape(np.prod(shape).item(), dim)
        outputs = self._forward(hidden_states)
        outputs = outputs.reshape(*shape, dim)
        return outputs

class Linear(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_size):
        return self.layer(hidden_size)

class Hashing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_table = config.ffn_num_table
        self.table_size = config.ffn_table_size
        self.total_dim = self.num_table * int(math.log2(self.table_size))
        self.num_repeat = int(math.ceil(self.total_dim / self.hidden_size))
        self.projections = nn.ModuleList([import_from_string(config.hash_linear_type)(config) for _ in range(self.num_repeat)])

    def forward(self, hidden_state):
        batch_size = hidden_state.shape[0]
        outputs = []
        for projection in self.projections:
            outputs.append(projection(hidden_state))
        outputs = torch.cat(outputs, dim = -1)
        if outputs.shape[-1] != self.total_dim:
            outputs = outputs[:, :self.total_dim].contiguous()
        outputs = outputs.reshape(batch_size, self.num_table, int(math.log2(self.table_size)))
        return outputs

class SoftmaxAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_head_size = config.attention_head_size

        if config.attention_probs_dropout_prob != 0:
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        else:
            self.dropout = nn.Identity()

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        attention_mask = -1000.0 * (1.0 - attention_mask[:, None, None, :].float())
        scale = math.sqrt(math.sqrt(self.attention_head_size))
        query_layer = query_layer.float() / scale
        key_layer = key_layer.float() / scale
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores + attention_mask
        attention_probs = self.dropout(nn.functional.softmax(attention_scores, dim = -1))
        context_layer = torch.matmul(attention_probs, value_layer.float()).to(value_layer.dtype)
        return context_layer

class LookupAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_query_per_table = config.attn_num_query_per_table
        self.num_query_per_table_val = config.attn_num_query_per_table_val
        self.compute_type = config.attn_compute_type
        self.num_table = config.attn_num_table
        self.table_size = config.attn_table_size

        self.code_length = int(math.log2(self.table_size))
        self.num_head = config.num_attention_heads
        self.head_dim = config.attention_head_size

    def extra_repr(self):
        strs = []
        strs.append(f"random, num_table={self.num_table}, table_size={self.table_size}")
        strs.append(f"num_query_per_table={self.num_query_per_table}, compute_type={self.compute_type}")
        return "\n".join(strs)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        batch_size, num_heads, sequence_length, head_dim = query_layer.shape
        projections = torch.randn(self.num_head, self.num_table, self.code_length, self.head_dim, dtype = query_layer.dtype, device = query_layer.device) / math.sqrt(self.head_dim)
        q_hash_scores = torch.einsum("bhnd,htcd->bhntc", query_layer, projections)
        k_hash_scores = torch.einsum("bhnd,htcd->bhntc", key_layer, projections)
        value_layer = value_layer * attention_mask[:, None, :, None]

        q_hash_scores = q_hash_scores.reshape(batch_size * num_heads, sequence_length, self.num_table, self.code_length).contiguous()
        k_hash_scores = k_hash_scores.reshape(batch_size * num_heads, sequence_length, self.num_table, self.code_length).contiguous()
        value_layer = value_layer.reshape(batch_size * num_heads, sequence_length, head_dim).contiguous()

        tables = construct_tables(q_hash_scores, value_layer, self.num_query_per_table, self.compute_type)
        outputs = query_tables(k_hash_scores, tables, self.num_query_per_table_val, self.compute_type)

        context_layer = outputs.reshape(batch_size, num_heads, sequence_length, head_dim)

        return context_layer

class YOSOAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hashcode_len = config.attention_hashcode_len
        self.num_hash_f = config.attention_num_hash_f
        self.num_head = config.num_attention_heads

        self.use_conv = hasattr(config, "attention_conv_kernel_size")
        if self.use_conv:
            self.conv = nn.Conv2d(
                in_channels = self.num_head, out_channels = self.num_head,
                kernel_size = (config.attention_conv_kernel_size, 1), padding = (config.attention_conv_kernel_size // 2, 0),
                bias = False,
                groups = self.num_head)

    def forward(self, Q, K, V, mask):
        if self.use_conv:
            conv_V = self.conv(V * mask[:, None, :, None])

        batch_size, num_heads, seq_len, head_dim = Q.size()

        Q = Q.reshape(batch_size * num_heads, seq_len, head_dim)
        K = K.reshape(batch_size * num_heads, seq_len, head_dim)
        V = V.reshape(batch_size * num_heads, seq_len, head_dim)

        mask = mask.int()[:, None, :].repeat(1, num_heads, 1).reshape(batch_size * num_heads, seq_len)

        if self.num_hash_f < 0:
            Q = yoso.normalize(Q)
            K = yoso.normalize(K)
            X = yoso.yoso_e(Q, K, V, mask, mask, self.hashcode_len)
        else:
            if self.training:
                Q = yoso.normalize(Q)
                K = yoso.normalize(K)
            X = yoso.yoso(
                Q.float().contiguous(), K.float().contiguous(), V.float().contiguous(),
                mask.contiguous(), mask.contiguous(),
                self.num_hash_f, self.hashcode_len
            ).to(Q.dtype)

        X = yoso.normalize(X)

        X = X.reshape(batch_size, num_heads, seq_len, head_dim)

        if self.use_conv:
            X += conv_V

        return X

    def extra_repr(self):
        return f'num_hash_f={self.num_hash_f}, hashcode_len={self.hashcode_len}, use_conv={self.use_conv}'

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if not hasattr(config, "attention_type"):
            self.attention = SoftmaxAttention(config)
        else:
            self.attention = import_from_string(config.attention_type)(config)

        self.checkpoint_attention = False
        if hasattr(config, "checkpoint_attention"):
            self.checkpoint_attention = config.checkpoint_attention

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)

        self.query = import_from_string(config.attntion_linear_type)(config)
        self.key = import_from_string(config.attntion_linear_type)(config)
        self.value = import_from_string(config.attntion_linear_type)(config)
        self.dense = import_from_string(config.attntion_linear_type)(config)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        hidden_states = self.LayerNorm(hidden_states)

        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        if self.checkpoint_attention:
            context_layer = checkpoint(self.attention, query_layer, key_layer, value_layer, attention_mask)
        else:
            context_layer = self.attention(query_layer, key_layer, value_layer, attention_mask)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        attention_output = self.dense(context_layer)
        return attention_output

class LookupFFN(nn.Module):
    def __init__(self, config, output_size):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.output_size = output_size
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps = config.layer_norm_eps)

        self.num_query_per_table = config.ffn_num_query_per_table
        self.num_query_per_table_val = config.ffn_num_query_per_table_val
        self.compute_type = config.ffn_compute_type
        self.num_table = config.ffn_num_table
        self.table_size = config.ffn_table_size
        assert (2 ** int(math.log2(self.table_size))) == self.table_size
        self.hash = Hashing(config)
        self.tables_weight = nn.Parameter(0.02 * torch.randn(self.num_table, self.table_size, self.output_size))
        self.tables_bias = nn.Parameter(0.001 * torch.randn(self.output_size))

    def extra_repr(self):
        strs = []
        strs.append(f"hidden_size={self.hidden_size}, num_table={self.num_table}, table_size={self.table_size}, output_size={self.output_size}")
        strs.append(f"num_query_per_table={self.num_query_per_table}, compute_type={self.compute_type}")
        return "\n".join(strs)

    def _forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        hidden_states = self.LayerNorm(hidden_states)

        hash_scores = self.hash(hidden_states)
        tables = self.tables_weight[None, :, :, :]
        if self.training:
            query_output = query_tables(hash_scores[None, :, :, :], tables, self.num_query_per_table, self.compute_type)
        else:
            query_output = query_tables(hash_scores[None, :, :, :], tables, self.num_query_per_table_val, self.compute_type)
        outputs = query_output + self.tables_bias

        return outputs

    def forward(self, hidden_states):
        shape = hidden_states.shape[:-1]
        dim = hidden_states.shape[-1]
        hidden_states = hidden_states.reshape(np.prod(shape).item(), dim)
        outputs = self._forward(hidden_states)
        outputs = outputs.reshape(*shape, self.output_size)
        return outputs

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.FFN = LookupFFN(config, self.hidden_size)
        self.attention = Attention(config)
        self.dropout = nn.Dropout(p = config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attn_output = self.dropout(self.attention(hidden_states, attention_mask)) + hidden_states
        layer_output = self.dropout(self.FFN(attn_output)) + attn_output
        return layer_output