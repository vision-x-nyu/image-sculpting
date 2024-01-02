import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor

import image_sculpting
from image_sculpting.systems.base import BaseDiffusionSystem
from threestudio.utils.typing import *

@image_sculpting.register("ours-system")
class OursSystem(BaseDiffusionSystem):
    def configure(self):
        super().configure()
        
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        pass

    def on_fit_start(self) -> None:
        super().on_fit_start()


    def training_step(self, batch, batch_idx):
        pass 

    def validation_step(self, batch, batch_idx):
        pass
        
    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        # render mesh
        out = self.renderer(**batch)
        background_img = F.interpolate(
            self.background_img,
            (out["color"].shape[2], out["color"].shape[3])
        ).float()
       
        masks = out["color"][:, 3:]
        image = (masks * out["color"][:, :3] + (1 - masks) * background_img).float()
        
        # Reset the forward method of SDXL to its original forward pass for DDIM inversion.
        self.model.reset_forward_methods()
        latents_inv = self.inversion.get_inverted_latent(
            pipe=self.model.pipe,
            color=image,
            background_img=background_img,
            depth=out["depth"],
            img=batch_idx,
        )
        # Overwrite the forward method of SDXL to enable injection
        self.model.overwrite_forward_methods()
        
        refine = self.model(
            depth=out["depth"],
            latents_inv=latents_inv,
            masks=masks,
            background_img=background_img,
            **batch
        )
        
        refine = refine.detach().cpu().permute([0, 2, 3, 1])
        self.save_image_grid(
            f"./{batch['index'][0]:04}.jpg",
           [
                {
                    "type": "rgb",
                    "img": refine[0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            
            ,
            name="test_step",
            step=self.true_global_step,
        )
       
    def on_test_epoch_end(self):
        pass
       
