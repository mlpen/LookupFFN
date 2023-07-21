import torch.nn as nn
import torch
import math
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import numpy as np
from src.args import import_from_string
import time

from ..lsh_layers.mongoose_slide.slide_lib.mongoose_network import LSHSampledLayer

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.checkpoint_attention = False
        if hasattr(config, "checkpoint_attention"):
            self.checkpoint_attention = config.checkpoint_attention

        if config.attention_probs_dropout_prob != 0:
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        else:
            self.dropout = nn.Identity()

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

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

class SlideFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_tokens_per_pass = config.num_tokens_per_pass
        self.hidden_size = config.hidden_size
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps = config.layer_norm_eps)

        self.lsh_layer = LSHSampledLayer(hash_weight = None, layer_size = config.hidden_size, K = config.num_hash_funcs, 
            L = config.num_hash_tables, num_class = config.intermediate_size, rehash_all = True)
        self.act = nn.GELU()
        
        k = math.sqrt(1 / config.intermediate_size)
        self.weights = nn.Parameter(torch.FloatTensor(config.intermediate_size, config.hidden_size).uniform_(-k, k))
        self.bias = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_(-k, k))

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.last_print_time = time.time()

    def _forward(self, hidden_states):
        normed_states = self.LayerNorm(hidden_states)
        self.lsh_layer.rehash()
        sizes = []
        outputs = []
        triplet_losses = []
        for b in range(0, normed_states.shape[0], self.num_tokens_per_pass):
            hidden_states = normed_states[b:(b + self.num_tokens_per_pass)]

            sample_logits, sample_ids, triplet_loss = self.lsh_layer(hidden_states)
            sample_logits = self.act(sample_logits)
            # if True: #time.time() - self.last_print_time > 10:
            #     self.last_print_time = time.time()
            #     print(sample_ids.shape[0])
            sizes.append(sample_ids.shape[0])
            sample_weights = F.embedding(sample_ids, self.weights, sparse = False) 

            dense_product = sample_logits.matmul(sample_weights)
            dense_logits = dense_product + self.bias

            outputs.append(self.dropout(dense_logits))
            triplet_losses.append(triplet_loss)
        print(sum(sizes) / len(sizes))
        outputs = torch.cat(outputs, dim = 0)
        triplet_loss = torch.stack(triplet_losses).mean()
        return outputs, triplet_loss

    def forward(self, hidden_states):
        shape = hidden_states.shape[:-1]
        dim = hidden_states.shape[-1]
        hidden_states = hidden_states.reshape(np.prod(shape).item(), dim)
        outputs, triplet_loss = self._forward(hidden_states)
        outputs = outputs.reshape(*shape, self.hidden_size)
        return outputs, triplet_loss

class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.FFN = SlideFFN(config)
        self.attention = Attention(config)
        self.dropout = nn.Dropout(p = config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attn_output = self.dropout(self.attention(hidden_states, attention_mask)) + hidden_states
        layer_output, triplet_loss = self.FFN(attn_output)
        layer_output = self.dropout(layer_output) + attn_output
        return layer_output, triplet_loss
