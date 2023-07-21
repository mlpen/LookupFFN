import torch.nn as nn
import torch
import math
from torch.utils.checkpoint import checkpoint
from ..postnorm import RobertaEmbeddings
import torch.nn.functional as F
from .layer import EncoderLayer

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

class SlideModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = Encoder(config)

    def forward(self, input_ids, token_type_ids, position_ids, attention_mask, **kwargs):
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        sequence_output = self.encoder(embedding_output, attention_mask)
        extra_outputs = {}
        return sequence_output, extra_outputs
