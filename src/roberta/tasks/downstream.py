import pytorch_lightning as pl
import torch
import os
import json
import time
from src.base_model_module import BaseModelModule
from collections import defaultdict

class DownstreamModelModule(BaseModelModule):
    def __init__(self, config, data_module):
        super().__init__(config, data_module)
        self.best_metrics = {}

    def validation_epoch_end(self, outputs):

        summary = defaultdict(list)
        if isinstance(outputs, list):
            for idx in range(len(outputs)):
                for output in outputs[idx]:
                    for key, val in self.sync_dict(output).items():
                        summary[f"{idx}-{key}"].append(val.item())
        else:
            for output in outputs:
                for key, val in self.sync_dict(output).items():
                    summary[key].append(val.item())
                    
        summary = {key:sum(val)/float(len(val)) for key, val in summary.items()}
        for key in summary:
            if f"{key}-low" not in self.best_metrics:
                self.best_metrics[f"{key}-low"] = summary[key]
                self.best_metrics[f"{key}-high"] = summary[key]
            else:
                self.best_metrics[f"{key}-low"] = min(self.best_metrics[f"{key}-low"], summary[key])
                self.best_metrics[f"{key}-high"] = max(self.best_metrics[f"{key}-high"], summary[key])

        for key, val in self.best_metrics.items():
            self.log(f"val.best.{key}", val, prog_bar = True, logger = True)
