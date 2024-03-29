import torch.nn as nn
import torch
import math
from torch.utils.checkpoint import checkpoint
from .postnorm import RobertaEmbeddings
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.checkpoint_attention = False
        if hasattr(config, "checkpoint_attention"):
            self.checkpoint_attention = config.checkpoint_attention

        if config.attention_probs_dropout_prob != 0:
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        else:
            self.dropout = nn.Identity()

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dense = nn.Linear(self.all_head_size, config.hidden_size)

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

    def attention(self, query_layer, key_layer, value_layer, attention_mask):
        attention_mask = -1000.0 * (1.0 - attention_mask[:, None, None, :].float())
        scale = math.sqrt(math.sqrt(self.attention_head_size))
        query_layer = query_layer.float() / scale
        key_layer = key_layer.float() / scale
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores + attention_mask
        attention_probs = self.dropout(nn.functional.softmax(attention_scores, dim = -1))
        context_layer = torch.matmul(attention_probs, value_layer.float()).to(value_layer.dtype)
        return context_layer

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = Attention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.FFN = nn.Sequential(
            nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps),
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
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

class PrenormRobertaModel(nn.Module):
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
