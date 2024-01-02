import argparse
import contextlib
import logging
import os
import sys
import pytorch_lightning as pl
import torch
from loguru import logger as lg 

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from threestudio.utils.misc import ColoredFilter
from threestudio.utils.callbacks import (
        CodeSnapshotCallback,
        ConfigSnapshotCallback,
        CustomProgressBar,
        ProgressCallback,
    )
lg.remove()  # Remove the default handler

# Add a new handler with green-colored log text
green_color_start = "\033[92m"
green_color_end = "\033[0m"
lg.add(
    sys.stdout,
    format=f"{green_color_start}{{message}}{green_color_end}",
    level="INFO",
)

@lg.catch
def main(args, extras) -> None:
    env_gpus_str = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    env_gpus = list(env_gpus_str.split(",")) if env_gpus_str else []
    selected_gpus = [0]

    # Always rely on CUDA_VISIBLE_DEVICES if specific GPU ID(s) are specified.
    # As far as Pytorch Lightning is concerned, we always use all available GPUs
    # (possibly filtered by CUDA_VISIBLE_DEVICES).
    devices = -1
    if len(env_gpus) > 0:
        # CUDA_VISIBLE_DEVICES was set already, e.g. within SLURM srun or higher-level script.
        n_gpus = len(env_gpus)
    else:
        selected_gpus = list(args.gpu.split(","))
        n_gpus = len(selected_gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    import image_sculpting
    from image_sculpting.systems.base import BaseSystem
    from threestudio.utils.callbacks import (
        CustomProgressBar,
        ProgressCallback,
    )
    from image_sculpting.utils.config import ExperimentConfig, load_config

    #from threestudio.utils.config import load_config
    from threestudio.utils.misc import get_rank
    from threestudio.utils.typing import Optional

    logger = logging.getLogger("pytorch_lightning")
    
    for handler in logger.handlers:
        if handler.stream == sys.stderr:  # type: ignore
            handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
            handler.addFilter(ColoredFilter())
         
    # parse YAML config to OmegaConf
    cfg: ExperimentConfig
    cfg = load_config(args.config, cli_args=extras, n_gpus=n_gpus)
    
    # set a different seed for each device
    pl.seed_everything(cfg.seed + get_rank(), workers=True) 
    dm = image_sculpting.find(cfg.data_type)(cfg.data)
   
    system: BaseSystem = image_sculpting.find(cfg.system_type)(
        cfg.system
    )
  
    #system.set_save_dir(os.path.join(cfg.trial_dir, "save"))
    system.set_save_dir(cfg.trial_dir)

    callbacks = []
    if args.train:
        
        callbacks += [CustomProgressBar(refresh_rate=1)]

    def write_to_text(file, lines):
        with open(file, "w") as f:
            for line in lines:
                f.write(line + "\n")

    loggers = []
    if args.train:
        # make tensorboard logging dir to suppress warning
        rank_zero_only(
            lambda: os.makedirs(os.path.join(cfg.trial_dir, "tb_logs"), exist_ok=True)
        )()
        loggers += [
            TensorBoardLogger(cfg.trial_dir, name="tb_logs"),
        ] + system.get_loggers()
        rank_zero_only(
            lambda: write_to_text(
                os.path.join(cfg.trial_dir, "cmd.txt"),
                ["python " + " ".join(sys.argv), str(args)],
            )
        )()
    
    trainer = Trainer(
        callbacks=callbacks,
        logger=loggers,
        inference_mode=False,
        accelerator="gpu",
        devices=devices,
        **cfg.trainer,
    )
    def set_system_status(system: BaseSystem, ckpt_path: Optional[str]):
        if ckpt_path is None:
            return
        ckpt = torch.load(ckpt_path, map_location="cpu")
        system.set_resume_status(ckpt["epoch"], ckpt["global_step"])
   
    if args.train:
        trainer.fit(system, datamodule=dm)
        trainer.test(system, datamodule=dm)
    elif args.validate:
        trainer.validate(system, datamodule=dm)
    elif args.test:
        set_system_status(system, cfg.resume)
        trainer.test(system, datamodule=dm, ckpt_path=cfg.resume)
   
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config file")
    parser.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to be used. 0 means use the 1st available GPU. "
        "1,2 means use the 2nd and 3rd available GPU. "
        "If CUDA_VISIBLE_DEVICES is set before calling `launch.py`, "
        "this argument is ignored and all available GPUs are always used.",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--validate", action="store_true")
    group.add_argument("--test", action="store_true")

    
    parser.add_argument(
        "--typecheck",
        action="store_true",
        help="whether to enable dynamic type checking",
    )

    args, extras = parser.parse_known_args()
    main(args, extras)
