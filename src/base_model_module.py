import pytorch_lightning as pl
import torch
from dataclasses import dataclass, field, asdict
import os
import json
import time
from transformers import AutoConfig, PretrainedConfig
from transformers.trainer_pt_utils import get_parameter_names
from torch.optim import Adam, AdamW
from transformers.optimization import get_scheduler
import torch.optim

def get_optimizer(
    optimizer, optim_groups, base_learning_rate,
    adam_w_mode = True, adam_beta1 = 0.9, adam_beta2 = 0.98, adam_epsilon = 1e-5, **kwargs
):
    optimizer = optimizer.lower()
    optim_cls = {
        "adam": AdamW if adam_w_mode else Adam,
    }[optimizer]

    args = [optim_groups]
    kwargs = {
        "lr": base_learning_rate,
        "eps": adam_epsilon,
        "betas": (adam_beta1, adam_beta2),
    }
    if optimizer in {"fusedadam", "fusedlamb"}:
        kwargs["adam_w_mode"] = adam_w_mode
    optimizer = optim_cls(*args, **kwargs)

    return optimizer

class BaseModelModule(pl.LightningModule):
    def __init__(self, config, data_module):
        super().__init__()
        self.config = config
        if hasattr(self.config.model, "pretrained_config"):
            self.model_config = AutoConfig.from_pretrained(self.config.model.pretrained_config)
        else:
            self.model_config = PretrainedConfig()
        for key, val in self.config.model.to_dict().items():
            setattr(self.model_config, key, val)
        print(self.model_config)

        self.occupy_all_memory_flag = True
        if hasattr(self.config, "occupy_all_memory"):
            self.occupy_all_memory_flag = self.config.occupy_all_memory
        self.called_occupy_all_memory = False

    def try_occupy_all_memory(self):
        if not self.occupy_all_memory_flag or self.called_occupy_all_memory:
            return
        self.called_occupy_all_memory = True
        tmp_list = []
        while True:
            try:
                tmp_list.append(torch.ones(1024, 1024, 512, dtype = torch.float32, device = self.model.device))
            except Exception as e:
                print(e)
                break
        for tensor in tmp_list:
            del tensor
        del tmp_list

    def training_step(self, batch, batch_idx):
        self.try_occupy_all_memory()
        output = self.model(**batch)
        for key, val in self.sync_dict(output).items():
            self.log(f"train.{key}", val.item(), on_step = True, on_epoch = True, prog_bar = True, logger = True)
        return output

    def validation_step(self, batch, batch_idx, dataloader_idx = 0):
        output = self.model(**batch)
        for key, val in self.sync_dict(output).items():
            self.log(f"val.{key}", val.item(), on_step = True, on_epoch = True, prog_bar = True, logger = True)
        return output

    def log(self, *args, **kwargs):
        if self.trainer is None:
            return
        else:
            return super().log(*args, **kwargs)

    def get_world_size(self):
        if self.trainer is None:
            return 1
        else:
            return self.trainer.world_size

    def sync_dict(self, inp):
        world_size = self.get_world_size()
        out = {key:val.detach() / world_size for key, val in inp.items()}

        if self.get_world_size() == 1:
            return out

        for key in out:
            torch.distributed.all_reduce(out[key])
        return out

    def on_save_checkpoint(self, checkpoint):
        save_to_hf = self.config.save_to_hf if hasattr(self.config, "save_to_hf") else False
        if save_to_hf:
            path = os.path.join(self.config.save_dir_path, "hf_ckpts", f"epoch={self.current_epoch:05d}-step={self.global_step:08d}")
            self.model.save_pretrained(path)

    def configure_optimizers(self):

        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        params_decay = [p for n, p in self.named_parameters() if any(nd in n for nd in decay_parameters)]
        params_nodecay = [p for n, p in self.named_parameters() if not any(nd in n for nd in decay_parameters)]

        optim_groups = [
            {"params": params_decay, "weight_decay": self.config.optimizer.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = get_optimizer(optim_groups = optim_groups, **self.config.optimizer.to_dict())

        max_steps = self.trainer.max_steps
        if max_steps == -1:
            max_steps = self.trainer.estimated_stepping_batches
            print(f"Inferring max_steps: {max_steps}")

        scheduler = get_scheduler(
            self.config.optimizer.lr_scheduler_type,
            optimizer,
            num_warmup_steps = self.config.optimizer.warmup_steps,
            num_training_steps = max_steps,
        )

        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "reduce_on_plateau": False,
                    "monitor": "loss",
                }
            ],
        )
