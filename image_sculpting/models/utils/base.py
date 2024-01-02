from threestudio.utils.typing import *

from skimage.transform import resize


import torch
from torchvision import transforms

class BaseSD:
    @property
    def unet(self):
        return self.pipe.unet
    
    @property
    def controlnet(self):
        return self.pipe.controlnet
    
    @property
    def scheduler(self):
        return self.pipe.scheduler
    
    @property
    def vae(self):
        return self.pipe.vae 
    
    @property
    def tokenizer(self):
        return self.pipe.tokenizer
    
    @property
    def text_encoder(self):
        return self.pipe.text_encoder
    
    @property
    def text_encoder_2(self):
        return self.pipe.text_encoder_2

    def load_img(
        self, 
        image_path,
    ):
        image_pil = transforms.Resize((self.cfg.height, self.cfg.width))(
            Image.open(image_path).convert("RGB")
        )
        image = transforms.ToTensor()(image_pil)[None].to(self.device)
        return image

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def _encode_images(
            self, 
            vae,
            imgs: Float[Tensor, "B 3 H W"],
            weights_dtype,
        ) -> Float[Tensor, "B 4 64 64"]:
        imgs = imgs * 2.0 - 1.0
        posterior = vae.encode(imgs.to(weights_dtype)).latent_dist
        latents = posterior.sample() * vae.config.scaling_factor
        return latents.to(weights_dtype)

    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def _decode_latents(
        self, 
        vae,
        latents: torch.Tensor,
    ) -> Float[Tensor, "B 3 H W"]:  
        
        latents = latents / vae.config.scaling_factor
        imgs = (vae.decode(latents).sample / 2 + 0.5).clamp(0, 1)
        return imgs