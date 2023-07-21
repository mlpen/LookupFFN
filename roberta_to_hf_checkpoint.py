
import os
from src.args import Options, Args
from src.args import import_config
import copy
import sys
import json
import argparse
import pickle
from src.args import import_from_string
from transformers import PretrainedConfig, AutoConfig
import pytorch_lightning as pl
import torch
import os
import json
import time
from transformers import RobertaPreTrainedModel
import torch.nn as nn
from src.roberta.tasks.metrics import Loss, Accuracy
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

        output = {
            "loss":mlm_loss, "mlm_loss":mlm_loss, "mlm_accu":mlm_accu, "mlm_count":mlm_count
        }

        return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, required = True)
    args = parser.parse_args()

    print(f"args: {args}")
    config = import_config(args.config)

    config.save_dir_path = os.path.join(config.output_dir, args.config)

    if "pretrained_config" in config.model.to_dict():
        model_config = AutoConfig.from_pretrained(config.model.pretrained_config)
    else:
        model_config = PretrainedConfig()
    for key, val in config.model.to_dict().items():
        setattr(model_config, key, val)
    print(model_config)

    print(os.listdir(config.save_dir_path))

    for file in os.listdir(config.save_dir_path):
        if not file.endswith(".ckpt"):
            continue

        model = RobertaForMLM(model_config)

        path = os.path.join(config.save_dir_path, file)
        weights = torch.load(path, map_location = 'cpu')["state_dict"]

        weights = {key[len('model.'):]:val for key, val in weights.items()}

        model.load_state_dict(weights, strict = True)
        print(f"Loaded {path}")

        path = os.path.join(config.save_dir_path, "hf_ckpts", file[:-len(".ckpt")])

        model.save_pretrained(path)

        del model

if __name__ == "__main__":
    main()
