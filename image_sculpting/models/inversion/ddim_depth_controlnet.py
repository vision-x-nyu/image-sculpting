from dataclasses import dataclass

import torch
from transformers import logging
from diffusers import (
    StableDiffusionXLPipeline,
    DDIMScheduler,
)
from PIL import Image

from threestudio.utils.typing import *
from threestudio.utils.base import (
    BaseObject,
    BaseModule,
)
import image_sculpting
from image_sculpting.models.inversion.base import BaseDDIM
logging.set_verbosity_error()

@image_sculpting.register("ddim-depth-controlnet")
class DDIMDepthControlNetInversion(BaseModule, BaseDDIM):
    """
    This class performs DDIM inversion with Depth ControlNet. 
        https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py
    """
    @dataclass
    class Config(BaseModule.Config):
        """
        Dataclass for configuration parameters of DDIMDepthControlNetInversion.

        Attributes:
            pretrained_model_name_or_path (str): Path for the pretrained model.
            half_precision_weights (bool): Flag to indicate the use of half precision weights.
            seed (int): Seed value for randomness control.
            num_inference_steps (int): Number of inference steps for the DDIM process.
            prompt (str): Text prompt.
            height (int): Height of the generated images.
            width (int): Width of the generated images.
            num_images_per_prompt (int): Number of images to generate per prompt.
            
            controlnet_conditioning_scale (float): 
                https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py#L1025
            
            control_guidance_start (float), control_guidance_end (float): 
                https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py#L1032
            
        """
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
        half_precision_weights: bool = True 
        seed: int = 0

        num_inference_steps: int = 100
        prompt: str = ""

        height: int = 1024
        width: int = 1024

        num_images_per_prompt: int  = 1
        controlnet_conditioning_scale: float = 0.5
        control_guidance_start: float = 0.0
        control_guidance_end: float = 1.0 

    def configure(self):
        """
        Configures the DDIM inversion module, setting up the device, weights, and scheduler.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
        
        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path, 
            subfolder="scheduler"
        )

    @property
    def controlnet(self):
        return self.pipe.controlnet

    @torch.no_grad()
    def _ddim_inversion(
        self, 
        prompt_embeds: torch.Tensor, 
        image: torch.Tensor,
        latents: torch.Tensor, 
        controlnet_keep: List,
        controlnet_conditioning_scale: int,
        added_cond_kwargs=None,
        img: str="",
    ) -> torch.Tensor:
        
        timesteps = reversed(self.scheduler.timesteps)
        
        latents_list = []
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            for i, t in enumerate(timesteps):
                controlnet_cond_scale = controlnet_conditioning_scale
                if isinstance(controlnet_cond_scale, list):
                    controlnet_cond_scale = controlnet_cond_scale[0]
                cond_scale = controlnet_cond_scale * controlnet_keep[i]

                control_model_input = latents
                controlnet_prompt_embeds = prompt_embeds
                controlnet_added_cond_kwargs = added_cond_kwargs
                
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=False,
                    added_cond_kwargs=controlnet_added_cond_kwargs,
                    return_dict=False,
                )
                noise_pred = self.unet(latents, 
                                t, 
                                encoder_hidden_states=prompt_embeds,
                                cross_attention_kwargs=None,
                                down_block_additional_residuals=down_block_res_samples,
                                mid_block_additional_residual=mid_block_res_sample,
                                added_cond_kwargs=added_cond_kwargs).sample
            
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                neg_alpha = self.scheduler.alphas_cumprod[0]

                alpha_prod_t_prev = (
                    self.scheduler.alphas_cumprod[timesteps[i - 1]]
                    if i > 0 else neg_alpha
                )
                
                mu = alpha_prod_t ** 0.5
                mu_prev = alpha_prod_t_prev ** 0.5
                sigma = (1 - alpha_prod_t) ** 0.5
                sigma_prev = (1 - alpha_prod_t_prev) ** 0.5

                pred_x0 = (latents - sigma_prev * noise_pred) / mu_prev
                latents = mu * pred_x0 + sigma * noise_pred
                latents_list.append(latents)
                
        return torch.stack(latents_list, 0)

    @torch.no_grad()
    def get_inverted_latent(
        self,
        pipe: StableDiffusionXLPipeline, 
        color: Float[Tensor, "B 4 H W"],
        background_img: Float[Tensor, "B 3 H W"],
        depth: Image, 
        img: int,
    ) -> Float[Tensor, "T B 4 H W"]:
        """
        Performs the DDIM inversion process using controlnet.
        Returns:
            torch.Tensor: The tensor of inverted latent representations for every timestep.
        """
        self.pipe = pipe 
     
        image = color[:, :3]
        if self.cfg.prompt is not None and isinstance(self.cfg.prompt, str):
            batch_size = 1
        elif self.cfg.prompt is not None and isinstance(self.cfg.prompt, list):
            batch_size = len(self.cfg.prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        self.scheduler.set_timesteps(self.cfg.num_inference_steps)
        latents = self._encode_imgs(image)
        
        height, width = image.shape[2:]
        prompt_embeds, _, pooled_prompt_embeds, _ = self.pipe.encode_prompt(
            self.cfg.prompt, 
            device=self.device, 
            do_classifier_free_guidance=False
        )
        depth = self.pipe.prepare_image(
                image=depth,
                width=width,
                height=height,
                batch_size=batch_size * self.cfg.num_images_per_prompt,
                num_images_per_prompt=self.cfg.num_images_per_prompt,
                device=self.device,
                dtype=self.controlnet.dtype,
                do_classifier_free_guidance=False,
            )
        
        (control_guidance_start, 
            control_guidance_end) = [self.cfg.control_guidance_start], [self.cfg.control_guidance_end]
        controlnet_keep = []
        timesteps = self.scheduler.timesteps
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0])

        size = (height, width)
        add_time_ids = self.pipe._get_add_time_ids(
            size, (0, 0), size, dtype=prompt_embeds.dtype, 
            text_encoder_projection_dim=self.pipe.text_encoder_2.config.projection_dim
        ).to(self.device)
       
        added_cond_kwargs = {
            "text_embeds": pooled_prompt_embeds, 
            "time_ids": add_time_ids
        }
        
        out = self._ddim_inversion(
            prompt_embeds=prompt_embeds, 
            image=depth,
            latents=latents, 
            controlnet_keep=controlnet_keep,
            controlnet_conditioning_scale=self.cfg.controlnet_conditioning_scale,
            added_cond_kwargs=added_cond_kwargs,
            img=str(img),
        )
      
        return out