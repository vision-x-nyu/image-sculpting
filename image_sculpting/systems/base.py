from dataclasses import dataclass, field

import torch 

import image_sculpting
import threestudio
from threestudio.utils.typing import *
from threestudio.systems.base import BaseSystem

class BaseDiffusionSystem(BaseSystem):
    @dataclass
    class Config(BaseSystem.Config):
        model_type: str = ""
        model: dict = field(default_factory=dict)
        
        inpainting_type: Optional[str] = ""
        inpainting: Optional[dict] = field(default_factory=dict)

        inversion_type: Optional[str] = ""
        inversion: Optional[dict] = field(default_factory=dict)

        renderer_type: Optional[str] = ""
        renderer: Optional[dict] = field(default_factory=dict)

        pose_type: Optional[str] = ""
        pose: Optional[dict] = field(default_factory=dict)

    cfg: Config
    def configure(self) -> None:
        if self.cfg.inpainting:
            self.inpainting = image_sculpting.find(self.cfg.inpainting_type)(
                self.cfg.inpainting
            )
            self.background_img = self.inpainting()
            del self.inpainting
            torch.cuda.empty_cache()

        if self.cfg.inversion_type:
            self.inversion = image_sculpting.find(self.cfg.inversion_type)(
                self.cfg.inversion
            )

        if self.cfg.renderer_type:
            self.renderer = image_sculpting.find(self.cfg.renderer_type)(
                self.cfg.renderer
            )

        if self.cfg.pose_type:
            self.pose = image_sculpting.find(self.cfg.pose_type)(
                self.cfg.pose
            )
        
        self.model = image_sculpting.find(self.cfg.model_type)(
            self.cfg.model 
        )
        
    def on_fit_start(self) -> None:
        if self._save_dir is not None:
            threestudio.info(f"Validation results will be saved to {self._save_dir}")
        else:
            threestudio.warn(
                f"Saving directory not set for the system, visualization results will not be saved"
            )

    def on_test_end(self) -> None:
        if self._save_dir is not None:
            threestudio.info(f"Test results saved to {self._save_dir}")

    def on_predict_start(self) -> None:
        pass 

    def predict_step(self, batch, batch_idx):
        pass 

    def on_predict_epoch_end(self) -> None:
        pass

    def on_predict_end(self) -> None:
        pass

    