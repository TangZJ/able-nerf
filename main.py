import hydra
from omegaconf import OmegaConf
from pdb import set_trace as st
import torch
import numpy as np
import random
import os
from config import Systemcfg
from hydra.core.config_store import ConfigStore
from datetime import datetime

from ablenerf_litsystem import LitSystem

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from random import randint


cs = ConfigStore.instance()
cs.store(name='systemconfig', node=Systemcfg )


@hydra.main(version_base='1.1', config_path="conf", config_name="defaults")
def main(cfg: Systemcfg) -> None:

    current_dir = os.getcwd()
    main_dir = hydra.utils.get_original_cwd()
    curdate  = datetime.now().strftime("%d-%m-%y")
    time = datetime.now().strftime("%H-%M-%S")

    anti_collison = str(randint(0,100)) # lazy hacking - for some reason non rank 0 with hydra might create the same dir.
    logdir = os.path.join(main_dir, 'outputs','logs', curdate, time, anti_collison)
    ckptdir = os.path.join(main_dir, 'outputs', 'ckpt', curdate, time, anti_collison)

    print(OmegaConf.to_yaml(cfg))
    
    if cfg.expt_settings.seed is not None:
        setup_seed(cfg.expt_settings.seed)
    system = LitSystem(cfg)

    logger = TensorBoardLogger(
        save_dir=logdir,
        name=cfg.expt_settings.exp_name,
        default_hp_metric=False
    )


    pbar = TQDMProgressBar(refresh_rate=1)
    ckpt_cb = ModelCheckpoint(dirpath=ckptdir,
                              save_last=True,
                              monitor='val/psnr',
                              mode='max',
                              save_top_k=2,
                              )

    callbacks = [ckpt_cb, pbar]
           
    trainer = pl.Trainer(
        #max_steps=cfg.optimizer.max_steps,
        max_epochs=30,
        #val_check_interval=cfg.val.check_interval,
        callbacks=callbacks,
        logger=logger,
        enable_model_summary=True,
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        num_sanity_val_steps=0,
        benchmark=True,
        profiler=None,
        #profiler="simple" if torch.cuda.device_count() == 1 else None,
        #strategy=DDPPlugin(find_unused_parameters=False) if torch.cuda.device_count() > 1 else None,
        strategy="ddp",
        limit_val_batches=cfg.val.limit_batch_size
    )

    trainer.fit(system, ckpt_path=None)


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    main()

