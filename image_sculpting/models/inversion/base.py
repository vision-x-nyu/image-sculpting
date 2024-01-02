import os
from threestudio.utils.typing import *

from transformers import logging
from PIL import Image
import torch
import torchvision.transforms as T


class BaseDDIM:
    @property
    def unet(self):
        return self.pipe.unet
    
    @property
    def vae(self):
        return self.pipe.vae
    
    @property
    def tokenizer(self):
        return self.pipe.tokenizer
    
    @property
    def text_encoder(self):
        return self.pipe.text_encoder
    
    @torch.no_grad()
    def _get_text_embeds(
        self, 
        prompt: str, 
        negative_prompt: str,
    ):
        def _embed_text(text, truncation=True):
            input_data = self.tokenizer(
                text, 
                padding='max_length', 
                max_length=self.tokenizer.model_max_length,
                truncation=truncation, 
                return_tensors='pt'
            )
            return self.text_encoder(input_data.input_ids.to(self.device))[0]

        return torch.cat([_embed_text(negative_prompt, False), _embed_text(prompt)])

    @torch.no_grad()
    def _decode_latents(
        self, 
        latents: torch.Tensor,
    ):  
        latents = latents.to(torch.float32)
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latents = latents / self.vae.config.scaling_factor
            imgs = (self.vae.decode(latents).sample / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def _encode_imgs(self, imgs):
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.mean * self.vae.config.scaling_factor
        return latents