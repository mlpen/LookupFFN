import torch.nn as nn
import torch
import math
import numpy as np
from .bh4.kernel import bh4
from .compute_code_score.kernel import compute_code_score
from .gather.kernel import gather
import time

class BH4(nn.Module):
    def __init__(self, in_dim, out_dim, block_size, decay_coeff = 0.7):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.block_size = block_size
        self.decay_coeff = decay_coeff

        self.padded_in_dim = int(2 ** math.ceil(math.log2(in_dim)))
        self.num_repeat = max(1, int(math.ceil(self.out_dim / self.padded_in_dim)))
        self.num_block = self.padded_in_dim // self.block_size
        
        coeff = math.sqrt(self.block_size * self.padded_in_dim)
        self.weight = nn.Parameter(torch.randn(self.num_repeat, 4, self.num_block, self.block_size, self.block_size) / coeff)
        self.bias = nn.Parameter(torch.zeros(self.out_dim))

    def extra_repr(self):
        return f"in_dim={self.in_dim}, out_dim={self.out_dim}, block_size={self.block_size}, decay_coeff={self.decay_coeff}"

    def _forward(self, x):
        
        batch_size, dim = x.shape
        if dim < self.padded_in_dim:
            padding_dim = self.padded_in_dim - dim
            x = torch.cat([x, torch.zeros(batch_size, padding_dim, dtype = x.dtype, device = x.device)], dim = -1).contiguous()

        x = self.decay_coeff * bh4(x, self.weight, training = self.training) + (1 - self.decay_coeff) * x.repeat(1, self.num_repeat)
        x = x[:, :self.out_dim].contiguous() + self.bias
        
        return x

    def forward(self, xs):
        shape = xs.shape[:-1]
        dim = xs.shape[-1]
        xs = xs.reshape(np.prod(shape).item(), dim)
        outputs = self._forward(xs)
        outputs = outputs.reshape(*shape, self.out_dim)
        return outputs

class Hashing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_table = config.ffn_num_table
        self.table_size = config.ffn_table_size
        self.code_length = int(math.log2(self.table_size))
        self.total_dim = self.num_table * self.code_length
        if config.projection_type == "bh4":
            self.projection = BH4(self.hidden_size, self.total_dim, block_size = config.block_size)
        elif config.projection_type == "dense":
            self.projection = nn.Linear(self.hidden_size, self.total_dim)
        else:
            raise NotImplementedError

    def forward(self, x):
        B, D = x.shape
        t0 = time.time()
        z = self.projection(x)
        t1 = time.time()
        
        z = z.reshape(B, self.num_table, self.code_length)
        code, score = compute_code_score(z, training = self.training)

        t2 = time.time()
        print("projection", t1 - t0)
        print("compute_code_score", t2 - t1)
        return code, score

class LookupFFN(nn.Module):
    def __init__(self, config, output_size):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.output_size = output_size
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps = config.layer_norm_eps)

        self.num_table = config.ffn_num_table
        self.table_size = config.ffn_table_size
        assert (2 ** int(math.log2(self.table_size))) == self.table_size
        self.hash = Hashing(config)
        self.tables = nn.Parameter(0.02 * torch.randn(self.num_table, self.table_size, self.output_size))
        self.bias = nn.Parameter(0.001 * torch.randn(self.output_size))

    def extra_repr(self):
        strs = []
        strs.append(f"hidden_size={self.hidden_size}, num_table={self.num_table}, table_size={self.table_size}, output_size={self.output_size}")
        strs.append(f"num_query_per_table={self.num_query_per_table}, compute_type={self.compute_type}")
        return "\n".join(strs)

    def _forward(self, hidden_states):
        hidden_states = self.LayerNorm(hidden_states)

        code, score = self.hash(hidden_states)
        
        outputs = self.gather(code, score, self.tables)
        
        outputs = outputs + self.bias

        return outputs

    def forward(self, hidden_states):
        shape = hidden_states.shape[:-1]
        dim = hidden_states.shape[-1]
        hidden_states = hidden_states.reshape(np.prod(shape).item(), dim)
        outputs = self._forward(hidden_states)
        outputs = outputs.reshape(*shape, self.output_size)
        return outputs
    
    def gather(self, indexes, weights, tables):
        B, N = indexes.shape
        num_table, table_size, vector_dim = tables.shape
        assert weights.shape[0] == B
        assert weights.shape[1] == N

        t0 = time.time()
        indexes = indexes + torch.arange(num_table, device = indexes.device, dtype = indexes.dtype)[None] * table_size
        tables = tables.reshape(num_table * table_size, vector_dim)
        outputs = gather(indexes, weights, tables, training = self.training)

        t1 = time.time()
        print("gather", t1 - t0)

        return outputs