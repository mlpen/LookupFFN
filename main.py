
import shutil
import os
import pytorch_lightning as pl
from src.args import import_config, import_from_string
import argparse
import datetime
import logging
import copy
import sys
import json
import torch
import random
import time
import logging
from pytorch_lightning.loggers import CSVLogger, WandbLogger

def main(config):

    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval = "step")]
    if config.save_top_k > 0:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                save_last = True,
                save_top_k = config.save_top_k,
                dirpath = config.save_dir_path,
                monitor = "step",
                mode = "max",
                filename = "{epoch:05d}-{step:08d}",
                save_on_train_epoch_end = False,
                every_n_epochs = 0 if config.save_top_k == 0 else None
            )
        )

    if config.test_run:
        trainer_logger = True
    else:
        trainer_logger = WandbLogger(project = config.project, name = config.run_name, save_dir = "..")

    trainer = pl.Trainer.from_argparse_args(
        config.trainer,
        replace_sampler_ddp = False,
        callbacks = callbacks,
        default_root_dir = config.save_dir_path if config.save_top_k > 0 else None,
        accelerator = 'cpu' if config.use_cpu else 'gpu',
        logger = trainer_logger
    )

    if not os.path.exists(config.save_dir_path) and trainer.global_rank == 0:
        os.makedirs(config.save_dir_path)

    if trainer.global_rank == 0:
        print(config)

    if os.path.exists(os.path.join(config.save_dir_path, "last.ckpt")):
        config.seed = config.seed * random.randrange(10000)
        print(f"new seed: {config.seed}")
    pl.utilities.seed.seed_everything(config.seed)

    # if trainer.global_rank == 0 and not config.test_run:
    #     # copying all implementations and config for reproducing results in case files are changed in the future.
    #     code_version_index = 0
    #     code_stored = False
    #     while not code_stored:
    #         code_folder = os.path.join(config.save_dir_path, f"code-version-{code_version_index}")
    #         if not os.path.exists(code_folder):
    #             try:
    #                 print("backup config and source code")
    #                 os.mkdir(code_folder)
    #                 with open(os.path.join(code_folder, "config.txt"), "w") as f:
    #                     f.write(str(config))
    #                 shutil.make_archive(f"{code_folder}/src", "zip", "src")
    #                 code_stored = True
    #             except Exception as e:
    #                 print(e)
    #         else:
    #             code_version_index += 1

    print(f"*********** data module set up ***********")
    data = import_from_string(config.data.pl_module)(config)
    data.setup()

    print(f"*********** model module set up ***********")
    model = import_from_string(config.model.pl_module)(config, data)

    if trainer.global_rank == 0:
        print(trainer)
        print(data)
        print(model)

    print(f"*********** start training ***********")
    possible_ckpt_path = os.path.join(config.save_dir_path, "last.ckpt")
    if os.path.exists(possible_ckpt_path):
        print(f"Resuming from checkpoint to {possible_ckpt_path}")
    elif hasattr(config, "resume_ckpt_path"):
        print(f"Resuming from checkpoint to {config.resume_ckpt_path}")
        possible_ckpt_path = config.resume_ckpt_path
    else:
        possible_ckpt_path = None

    if config.val_only:
        trainer.validate(model = model, datamodule = data, ckpt_path = possible_ckpt_path)
    else:
        trainer.fit(model = model, datamodule = data, ckpt_path = possible_ckpt_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type = str, required = True)
    parser.add_argument('--test_run', action = 'store_true')
    parser.add_argument('--val_only', action = 'store_true')
    parser.add_argument('--all_memory', action = 'store_true')
    parser.add_argument('--use_cpu', action = 'store_true')
    args = parser.parse_args()

    print(f"args: {args}")

    config = import_config(args.config)
    config.run_name = args.config.replace(os.sep, "-")
    if config.trainer.gpus == -1:
        config.trainer.gpus = torch.cuda.device_count()
    config.save_dir_path = os.path.join(config.output_dir, args.config)
    config.config_path = args.config

    if not hasattr(config, "test_run"):
        config.test_run = args.test_run
    if not hasattr(config, "val_only"):
        config.val_only = args.val_only
    if not hasattr(config, "occupy_all_memory"):
        config.occupy_all_memory = args.all_memory
    if not hasattr(config, "use_cpu"):
        config.use_cpu = args.use_cpu

    if config.test_run:
        config.trainer.gpus = 1
        config.optimizer.batch_size = min(2, config.optimizer.batch_size)

    main(config)
