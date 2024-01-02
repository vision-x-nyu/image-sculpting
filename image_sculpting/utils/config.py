import os
from dataclasses import dataclass, field
from datetime import datetime

from omegaconf import OmegaConf

import threestudio
from threestudio.utils.typing import *


@dataclass
class ExperimentConfig:
    name: str = ""
    exp_root_dir: str = ""
    seed: int = 0
    trial_dir: str = ""
    resume: Optional[str] = None
    data_type: str = None 
    data: dict = field(default_factory=dict)
    n_gpus: int = 1
   
    system_type: str = ""
    system: dict = field(default_factory=dict)

    trainer: Optional[dict] = field(default_factory=dict)

    def __post_init__(self):
        self.trial_dir = os.path.join(self.exp_root_dir, self.name)
        os.makedirs(self.trial_dir, exist_ok=True)

def load_config(*yamls: str, cli_args: list = [], from_string=False, **kwargs) -> Any:
    if from_string:
        yaml_confs = [OmegaConf.create(s) for s in yamls]
    else:
        yaml_confs = [OmegaConf.load(f) for f in yamls]
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*yaml_confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    scfg = parse_structured(ExperimentConfig, cfg)
    return scfg


def config_to_primitive(config, resolve: bool = True) -> Any:
    return OmegaConf.to_container(config, resolve=resolve)


def dump_config(path: str, config) -> None:
    with open(path, "w") as fp:
        OmegaConf.save(config=config, f=fp)

def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg