
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import pyarrow as pa
import glob
import logging
import os, sys, json
import numpy as np
import pathlib
import torch
import datasets
from .data_module_base import BaseDataModule, SubDataset

class DownstreamDataModule(BaseDataModule):

    def setup(self, stage = None):
        if stage is not None:
            print(f"Setting data module stage {stage} has no effect")

        self.training_dataset = datasets.load_from_disk(self.config.training_dataset_path)
        if isinstance(self.config.validation_dataset_path, list):
            self.validation_dataset = [
                datasets.load_from_disk(path)
                for path in self.config.validation_dataset_path
            ]
        else:
            self.validation_dataset = datasets.load_from_disk(self.config.validation_dataset_path)

    def get_indeces_subset(self, dataset_size):
        rank = self.trainer.global_rank
        world_size = self.trainer.world_size

        indices = np.arange(dataset_size, dtype = np.uint32)
        rng = np.random.default_rng(self.seed + self.trainer.current_epoch)
        rng.shuffle(indices)

        size_per_rank = dataset_size // world_size
        offset_start = rank * size_per_rank
        offset_end = (rank + 1) * size_per_rank

        indices = indices[offset_start:offset_end]
        return indices

    def train_dataloader(self):

        indices = self.get_indeces_subset(len(self.training_dataset))
        dl = DataLoader(
            SubDataset(self.training_dataset, indices),
            batch_size = self.batch_size,
            collate_fn = self.data_collator,
            num_workers = self.config.num_workers,
            drop_last = True,
            prefetch_factor = 4
        )

        return dl

    def val_dataloader(self):

        if isinstance(self.validation_dataset, list):
            dl = [
                DataLoader(
                    SubDataset(dataset, self.get_indeces_subset(len(dataset))),
                    batch_size = self.batch_size,
                    collate_fn = self.data_collator,
                    num_workers = self.config.num_workers,
                    drop_last = True,
                    prefetch_factor = 4
                )
                for dataset in self.validation_dataset
            ]
        else:
            indices = self.get_indeces_subset(len(self.validation_dataset))
            dl = DataLoader(
                SubDataset(self.validation_dataset, indices),
                batch_size = self.batch_size,
                collate_fn = self.data_collator,
                num_workers = self.config.num_workers,
                drop_last = True,
                prefetch_factor = 4
            )
        return dl
