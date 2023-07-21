
from transformers import AutoTokenizer
import pytorch_lightning as pl
import logging
import os, sys, json
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from src.args import import_from_string

class IndexDataset(Dataset):
    def __init__(self, dataset_indices):
        self.dataset_indices = dataset_indices

    def __getitem__(self, index):
        return self.dataset_indices[index]

    def __len__(self):
        return len(self.dataset_indices)

class SubDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        return self.dataset[self.indices[index].item()]

    def __len__(self):
        return len(self.indices)

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config.data
        self.seed = config.seed
        self.batch_size = config.optimizer.batch_size

        os.environ["TOKENIZERS_PARALLELISM"] = "False"
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        self.data_collator = import_from_string(self.config.collator)(tokenizer = self.tokenizer, **self.config.collator_args.to_dict())
