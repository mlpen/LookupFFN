import pytorch_lightning as pl
import torch
import torch.nn as nn
import os
import json
import time
from transformers import RobertaPreTrainedModel
from collections import defaultdict
from .downstream import DownstreamModelModule
from .metrics import Loss, Accuracy
from src.utils import filter_inputs
from src.args import import_from_string

class ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RobertaForSequenceClassificaiton(RobertaPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.roberta = import_from_string(config.encoder_type)(config)
        self.classifier = ClassificationHead(config)
        self.loss_fct = Loss()
        self.accu_fct = Accuracy()
        self.post_init()

    def forward(
        self,
        label,
        **kwargs
    ):
        outputs = self.roberta(**filter_inputs(self.roberta.forward, kwargs))
        sequence_output = outputs[0]

        scores = self.classifier(sequence_output[:, 0, :])
        loss, _ = self.loss_fct(scores, label)
        accu, count = self.accu_fct(scores, label)

        output = {
            "loss":loss, "accu":accu, "count":count
        }
        return output

class SequenceClassificaitonModelModule(DownstreamModelModule):
    def __init__(self, config, data_module):
        super().__init__(config, data_module)

        self.tokenizer = data_module.tokenizer
        self.model = RobertaForSequenceClassificaiton(self.model_config)
        self.model.resize_token_embeddings(len(self.tokenizer))

        if hasattr(self.config.model, "load_pretrain"):
            print(f"********* Loading pretrained weights: {self.config.model.load_pretrain}")
            checkpoint_model = import_from_string(self.config.model.load_pretrain["model_type"]).from_pretrained(self.config.model.load_pretrain["checkpoint"])
            checkpoint_model.resize_token_embeddings(len(self.tokenizer))
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_model.state_dict(), strict = False)
            print(f"missing_keys = {missing_keys}")
            print(f"unexpected_keys = {unexpected_keys}")
        else:
            print("********* Trained from scratch")
