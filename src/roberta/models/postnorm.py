import torch.nn as nn
import torch
import math
from torch.utils.checkpoint import checkpoint

class RobertaEmbeddings(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.pad_token_id = config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings + 2, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids, position_ids):

        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        position_embeddings = self.position_embeddings(self.pad_token_id + 1 + position_ids)

        embeddings = inputs_embeds + token_type_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

    def resize_position_embeddings(self, target_length, init_weight):
        target_length = target_length + 2
        old_length, old_embedding_dim = self.position_embeddings.weight.shape
        if old_length == target_length:
            return

        new_embeddings = nn.Embedding(target_length, old_embedding_dim)

        multipier = int(target_length / old_length + 2)
        new_embeddings_data = torch.cat(
            [self.position_embeddings.weight.data] +
            [self.position_embeddings.weight.data[2:, :]] * multipier
        , dim = 0)
        new_embeddings.weight.data[:target_length, :] = new_embeddings_data[:target_length, :].detach()
        self.position_embeddings = new_embeddings
        return self.position_embeddings

    def resize_token_type_embeddings(self, target_length, init_weight):
        old_length, old_embedding_dim = self.token_type_embeddings.weight.shape
        if old_length == target_length:
            return
        self.token_type_embeddings = nn.Embedding(target_length, old_embedding_dim)
        init_weight(self.token_type_embeddings)
        return self.token_type_embeddings


class RobertaSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.checkpoint_attention = config.checkpoint_attention if hasattr(config, "checkpoint_attention") else False
        if config.attention_probs_dropout_prob != 0:
            self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        else:
            self.dropout = nn.Identity()

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        attention_mask = -1000.0 * (1.0 - attention_mask[:, None, None, :].float())

        query_layer = self.transpose_for_scores(self.query(hidden_states)).float()
        key_layer = self.transpose_for_scores(self.key(hidden_states)).float()
        value_layer = self.transpose_for_scores(self.value(hidden_states)).float()

        if self.checkpoint_attention:
            context_layer = checkpoint(self.attention, query_layer, key_layer, value_layer, attention_mask).to(hidden_states.dtype)
        else:
            context_layer = self.attention(query_layer, key_layer, value_layer, attention_mask).to(hidden_states.dtype)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer

    def attention(self, query_layer, key_layer, value_layer, attention_mask):
        scale = math.sqrt(math.sqrt(self.attention_head_size))
        query_layer = query_layer / scale
        key_layer = key_layer / scale
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores + attention_mask
        attention_probs = self.dropout(nn.functional.softmax(attention_scores, dim = -1))
        context_layer = torch.matmul(attention_probs, value_layer)
        return context_layer

class RobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RobertaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = RobertaSelfAttention(config)
        self.output = RobertaSelfOutput(config)

    def forward(self, hidden_states, attention_mask):
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output

class RobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class RobertaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps = config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class RobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class RobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states

class RobertaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

    def forward(self, input_ids, token_type_ids, position_ids, attention_mask, **kwargs):
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        sequence_output = self.encoder(embedding_output, attention_mask)
        return sequence_output,

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
