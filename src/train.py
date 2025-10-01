
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CSVLogger


import yaml
import os
from parse import parse
from datetime import datetime

import models
import argparse
import pandas as pd
# from torch.nn import module_list
from datamodule import DataModule
from functools import partial
from loguru import logger
from utils import safe_open_yaml, stringify

current_directory = os.path.dirname(__file__)
python_file_name = os.path.splitext(os.path.basename(__file__))[0]

OUTPUT_DIR = "/results/" 
CONFIG_FILE = "/src/config/train_baseline.yaml"

def _init_model(model_name, model_params):
    model_cls       = getattr(models, model_name)
    model           = model_cls(**model_params)

    return model

def _validate(trainer, data_module, model_name, best_ckpt, model_params, save_dir, fast_dev_run=True):


    best_ckpt = ckpt_callback.best_model_path
    logger.info(f"The best model can be found in {best_ckpt}")
    model_cls       = getattr(models, model_name)
    best_model = model_cls.load_from_checkpoint(best_ckpt, **model_params)

    logger.info("Running validate with best model..")
    best_model.heavy_eval = True
    val_metrics = trainer.validate(best_model, datamodule=data_module)
    
    if fast_dev_run:
        output_yaml = f"{save_dir}/fast_dev_run.yaml"
    else:
        output_yaml = f"{save_dir}/output.yaml"
    
    logger.info("If you want to use the model with this validation result as the model for the evalAI test, run the following")
    logger.info(f"python save_for_submission.py -c {os.path.abspath(output_yaml)}.")
    logger.info(f"You don't have to run test as it will be done on our server.")
    payload = {
        "best_model_path": os.path.abspath(best_ckpt) or None,
        "metrics": val_metrics,
        "config": os.path.abspath(CONFIG_FILE)
    }
    with open(output_yaml, "w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


if __name__ == "__main__":

    config = safe_open_yaml(CONFIG_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    model_name      = config["model"]
    model_params    = config["model_params"]
    model_params_str = stringify(model_params, delimiter="_")
    save_dir                = f"{OUTPUT_DIR}/{model_name}_{model_params_str}"
    save_dir_fast_dev_run   = f"{OUTPUT_DIR}/{model_name}_{model_params_str}_fast_dev_run"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_fast_dev_run, exist_ok=True)

    model = _init_model(model_name, model_params)

    data_module = DataModule(**config["datamodule"])
    seed_everything(8, workers=True)

    ckpt_callback = ModelCheckpoint(
        monitor="val/loss", 
        filename='{epoch:03d}-{val/loss:.2f}',
        mode="min",
    )
    pl_logger = CSVLogger("logs", name=f"{model_name}_{model_params_str}_fast_dev_run")

    logger.info("Running fast development run with single epoch.")
    trainer = pl.Trainer(
        max_epochs=1,
        default_root_dir=save_dir_fast_dev_run,
        callbacks=[ckpt_callback],
        logger=pl_logger,
        gradient_clip_val=0.5,
        limit_train_batches=5,
        limit_val_batches=5,
        devices=1
    )

    trainer.fit(model, data_module)
    _validate(trainer, data_module, model_name, ckpt_callback.best_model_path, model_params, save_dir_fast_dev_run, fast_dev_run=True)
    ########## RUN ACTUAL #################
    
    ckpt_callback = ModelCheckpoint(
        monitor="val/loss", 
        save_top_k=5,
        filename='{epoch:03d}-{step}-{val/loss:.2f}',
        mode="min",
    )
    pl_logger = CSVLogger("logs", name=f"{model_name}_{model_params_str}")
    trainer = pl.Trainer(
        max_epochs=300,
        default_root_dir=save_dir,
        callbacks=[ckpt_callback],
        gradient_clip_val=0.5,
        logger=pl_logger,
        devices=1
    )

    model = _init_model(model_name, model_params)

    trainer.fit(model, data_module)
    _validate(trainer, data_module, model_name, ckpt_callback.best_model_path, model_params, save_dir, fast_dev_run=False)