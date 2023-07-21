import pytorch_lightning as pl
import torch
import os
import json
import time
from transformers import RobertaPreTrainedModel
import torch.nn as nn
from .metrics import Loss, Accuracy
from src.base_model_module import BaseModelModule
from src.utils import filter_inputs
from src.args import import_from_string

class LMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gelu = nn.GELU()
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features):
        x = self.dense(features)
        x = self.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x

    def _tie_weights(self):
        self.bias = self.decoder.bias

class RobertaForMLM(RobertaPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.config = config
        self.roberta = import_from_string(config.encoder_type)(config)
        self.lm_head = LMHead(config)
        self.loss_fct = Loss()
        self.accu_fct = Accuracy()
        self.post_init()

    def forward(self, labels, **kwargs):
        outputs = self.roberta(**filter_inputs(self.roberta.forward, kwargs))
        sequence_output = outputs[0]

        if "masked_token_indices" in kwargs:
            masked_token_indices = kwargs["masked_token_indices"]
            batch_indices = torch.arange(sequence_output.shape[0], device = sequence_output.device)[:, None]
            sequence_output = sequence_output[batch_indices, masked_token_indices, :]
            labels = labels[batch_indices, masked_token_indices]

        logits = self.lm_head(sequence_output)
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.reshape(-1)
        mlm_loss, _ = self.loss_fct(logits, labels)
        mlm_accu, mlm_count = self.accu_fct(logits, labels)

        if hasattr(self.config, "aux_loss_coeff"):
            triloss = sum(outputs[1])
            output = {
                "loss":mlm_loss + self.config.aux_loss_coeff * triloss, "mlm_loss":mlm_loss, "mlm_accu":mlm_accu, "mlm_count":mlm_count, "triloss":triloss
            }
        else:
            output = {
                "loss":mlm_loss, "mlm_loss":mlm_loss, "mlm_accu":mlm_accu, "mlm_count":mlm_count
            }

        return output

    def resize_position_embeddings(self, target_length):
        gathered = []
        for module in self.roberta.modules():
            if hasattr(module, "resize_position_embeddings"):
                gathered.append(module)
        for module in gathered:
            module.resize_position_embeddings(target_length, self._init_weights)

    def resize_token_type_embeddings(self, target_length):
        gathered = []
        for module in self.roberta.modules():
            if hasattr(module, "resize_token_type_embeddings"):
                gathered.append(module)
        for module in gathered:
            module.resize_token_type_embeddings(target_length, self._init_weights)

class MLMModelModule(BaseModelModule):
    def __init__(self, config, data_module):
        super().__init__(config, data_module)
        self.model = RobertaForMLM(self.model_config)
        self.tokenizer = data_module.tokenizer

        if hasattr(self.config.model, "load_pretrain"):
            print(f"********* Loading pretrained weights: {self.config.model.load_pretrain}")
            checkpoint_model = import_from_string(self.config.model.load_pretrain["model_type"]).from_pretrained(self.config.model.load_pretrain["checkpoint"])

            try:
                checkpoint_model.resize_token_embeddings(len(self.tokenizer))
            except Exception as e:
                print(e)
            try:
                checkpoint_model.resize_position_embeddings(self.config.model.max_position_embeddings)
            except Exception as e:
                print(e)
            try:
                checkpoint_model.resize_token_type_embeddings(self.config.model.type_vocab_size)
            except Exception as e:
                print(e)

            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_model.state_dict(), strict = False)
            print(f"missing_keys = {missing_keys}")
            print(f"unexpected_keys = {unexpected_keys}")
