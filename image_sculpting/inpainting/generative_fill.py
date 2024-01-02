from dataclasses import dataclass, field

import imageio as io 
from torchvision import transforms

import image_sculpting
from threestudio.utils.base import (
    BaseObject,
    BaseModule,
)
from threestudio.utils.typing import *


@image_sculpting.register("generative-fill")
class GenerativeFill(BaseModule):
    """
    read image from the generative fill output.
    """
    @dataclass
    class Config(BaseModule.Config):
        bg_path: str = ""
      
    cfg: Config 
    def configure(self):
        pass 

    def forward(
        self,
        **kwargs,
    ) -> Float[Tensor, "N H W 3"]:
        bg = io.imread(self.cfg.bg_path)
        out = transforms.ToTensor()(bg)[None].to(self.device)
        return out



        
