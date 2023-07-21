import torch.nn as nn
import torch
import math
from torch.utils.checkpoint import checkpoint
from .postnorm import RobertaEmbeddings
import torch.nn.functional as F
from .prenorm import Attention
import numpy as np

class ShuffleNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.intermediate_block_size = (config.intermediate_size // config.hidden_size) * config.block_size
        self.hidden_size = config.hidden_size
        self.num_block = config.hidden_size // config.block_size
        self.w1 = nn.Parameter(0.02 * torch.randn(self.num_block, self.intermediate_block_size, self.block_size))
        self.b1 = nn.Parameter(0.001 * torch.randn(config.intermediate_size))
        self.w2 = nn.Parameter(0.02 * torch.randn(self.num_block, self.block_size, self.intermediate_block_size))
        self.b2 = nn.Parameter(0.001 * torch.randn(config.hidden_size))

        self.act = nn.GELU()
        
    def extra_repr(self):
        strs = [f"block_size={self.block_size}"]
        return "\n".join(strs)

    def _forward(self, hidden_states):
        batch_size, dim = hidden_states.shape

        hidden_states = hidden_states.reshape(batch_size, self.num_block, self.block_size)
        hidden_states = torch.einsum("bni,noi->bno", hidden_states, self.w1)
        hidden_states = hidden_states.transpose(-1, -2)
        hidden_states = hidden_states.reshape(batch_size, self.num_block * self.intermediate_block_size)
        hidden_states = hidden_states + self.b1
        hidden_states = self.act(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, self.num_block, self.intermediate_block_size)
        hidden_states = torch.einsum("bni,noi->bno", hidden_states, self.w2)
        hidden_states = hidden_states.reshape(batch_size, self.num_block * self.block_size)
        hidden_states = hidden_states + self.b2
        
        return hidden_states


    def forward(self, hidden_states):
        shape = hidden_states.shape[:-1]
        dim = hidden_states.shape[-1]
        hidden_states = hidden_states.reshape(np.prod(shape).item(), dim)
        outputs = self._forward(hidden_states)
        outputs = outputs.reshape(*shape, dim)
        return outputs

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = Attention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.FFN = nn.Sequential(
            ShuffleNet(config),
            nn.Dropout(config.hidden_dropout_prob)
        )

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout(attention_output) + hidden_states
        layer_output = self.FFN(attention_output) + attention_output
        return layer_output

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class Shuffle(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = Encoder(config)

    def forward(self, input_ids, token_type_ids, position_ids, attention_mask, **kwargs):
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        sequence_output = self.encoder(embedding_output, attention_mask)
        return sequence_output,

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
