import bisect
import math
import os
from dataclasses import dataclass, field
import glob 
import json 

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank

from threestudio.utils.typing import *

import image_sculpting
@dataclass
class ObjectCentricConfig:
    height: int = 1024
    width: int = 1024
    batch_size: int = 1
    elevation_deg: float = 0.0
    camera_distance: float = 3.8
    fovy_deg: float = 20.0
    n_views: int = 12 
    fix_view: bool = False 

class ObjectCentricDataset(Dataset):
    """
    Preparing camera parameters for render
    """
    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.cfg = parse_structured(
            ObjectCentricConfig, cfg
        )
        
        fovy = torch.deg2rad(torch.tensor([self.cfg.fovy_deg]))
       
        focal_length = (
            0.5 * self.cfg.height / torch.tan(0.5 * fovy)
        )
        if not self.cfg.fix_view:
            self.mesh_azimuth = torch.linspace(0, 360.0, self.cfg.n_views + 1)[:self.cfg.n_views]
        else:
            self.mesh_azimuth = torch.zeros(self.cfg.n_views)
       
        self.intrinsics = torch.tensor([[focal_length[0], 0, self.cfg.height / 2],
                                    [0, focal_length[0], self.cfg.width / 2],
                                    [0, 0, 1]])
        
        azimuth_deg = torch.zeros((1))
        
        elevation_deg = torch.full_like(azimuth_deg, self.cfg.elevation_deg)
        camera_distances = torch.full_like(elevation_deg, self.cfg.camera_distance)
        
        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        center = torch.zeros_like(camera_positions)
        up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.cfg.batch_size, 1)

        fovy_deg = torch.full_like(
            elevation_deg, self.cfg.fovy_deg
        )
        fovy = fovy_deg * math.pi / 180
        light_positions = camera_positions

        lookat = F.normalize(center - camera_positions, dim=-1)
        right = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4 = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0
      
        self.c2w = c2w

    def __len__(self):
        return self.cfg.n_views

    def __getitem__(self, index):
        return {
            "index": index,
            "mesh_azimuth": self.mesh_azimuth[index],
            "c2w": self.c2w[0],
            "intrinsics": self.intrinsics,
        }
       
@image_sculpting.register("object-centric-datamodule")
class ObjectCentricDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = cfg 
        

    def setup(self, stage=None) -> None:
        self.test_dataset = ObjectCentricDataset(self.cfg)

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset, num_workers=0, batch_size=batch_size, collate_fn=collate_fn
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            collate_fn=self.train_dataset.collate,
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(self.test_dataset, batch_size=1)
