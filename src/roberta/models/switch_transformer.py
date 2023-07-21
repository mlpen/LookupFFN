import torch.nn as nn
import torch
import math
from torch.utils.checkpoint import checkpoint
from .postnorm import RobertaEmbeddings
from .prenorm import Attention
import torch.nn.functional as F
from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersDenseActDense, SwitchTransformersLayerNorm, SwitchTransformersTop1Router
from transformers.models.switch_transformers.configuration_switch_transformers import SwitchTransformersConfig

class SwitchTransformersSparseMLP(nn.Module):
    r"""
    Implementation of the Switch Transformers Sparse MLP module.
    """

    def __init__(self, config: SwitchTransformersConfig, expert_class: nn.Module = SwitchTransformersDenseActDense):
        super().__init__()
        # Step 1: Get the correct router according to its class
        self.router = SwitchTransformersTop1Router(config)

        # Step 2: Get the experts
        self.experts = nn.ModuleDict()
        for idx in range(config.num_experts):
            self.experts[f"expert_{idx}"] = expert_class(config)

    def forward(self, hidden_states):
        r"""
        Hold on, this will be slightly tricky to understand In the correct order, a MoE layer does the following:
        1- Gets the `router_mask` from the router. The shape of the mask is `(batch_size, sequence_length, num_expert)`
        and corresponds to the argmax of the `router_probs`. The probabilities are needed in the computation of the
        hidden states : they are broadcasted to the hidden states values (can be interpreted as a scaling factor).
        2- Dispatch the tokens to its associated experts. We do a classic for loop over the experts and assign for each
        expert the corresponding hidden states.
        """
        # Step 1: Get the router_mask from the router as wel as the probabilities
        router_mask, router_probs, router_logits = self.router(hidden_states)
        expert_index = torch.argmax(router_mask, dim=-1)

        # The routers introduced might not always map all the tokens, to a router, which means that some hidden states
        # can be unchanged from one layer to another. That is why the hidden states are cloned before updating only the seleced ones.

        next_states = hidden_states.clone()
        for idx, expert in enumerate(self.experts.values()):

            token_indices = router_mask[:, :, idx].bool()
            next_states[token_indices] = expert(hidden_states[token_indices]).to(next_states.dtype)

        hidden_states = router_probs * next_states
        return hidden_states, (router_logits, expert_index)


class SwitchTransformersLayerFF(nn.Module):
    r"""
    Switch Transformers Feed Forward layer module. This is a wrapper around the Mixture of Experts module.
    Parameters:
        config : ([`SwitchTransformersConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
        is_sparse (`bool`):
            Whether the MLP layer is a `Sparse` layer (contains a Mixture of Experts) or not
    """

    def __init__(self, config: SwitchTransformersConfig, is_sparse=False):
        super().__init__()
        self.is_sparse = is_sparse

        # Check if it is a sparse layer, if not then it is a dense layer
        if not self.is_sparse:
            self.mlp = SwitchTransformersDenseActDense(config)
        else:
            self.mlp = SwitchTransformersSparseMLP(config)

        self.layer_norm = SwitchTransformersLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, output_router_logits):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states)

        if isinstance(forwarded_states, tuple):
            forwarded_states, router_tuple = forwarded_states
        else:
            router_tuple = None

        output = hidden_states + self.dropout(forwarded_states)

        if output_router_logits and router_tuple is not None:
            output = (output, router_tuple)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = Attention(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        switch_config = SwitchTransformersConfig()
        switch_config.d_model = config.hidden_size
        switch_config.d_ff = config.intermediate_size
        switch_config.expert_capacity = 1024 * 32
        self.FFN = SwitchTransformersLayerFF(switch_config, is_sparse = True)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout(attention_output) + hidden_states
        layer_output = self.FFN(attention_output, output_router_logits = False) + attention_output
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

class Model(nn.Module):
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
