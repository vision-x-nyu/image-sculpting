from dataclasses import dataclass

import torch
from torch import nn 
from diffusers import (
    DiffusionPipeline,
)
from diffusers.image_processor import (
    PipelineImageInput, 
)
from diffusers.utils.torch_utils import randn_tensor

from threestudio.utils.base import BaseModule
from threestudio.utils.typing import *
from image_sculpting.models.utils.base import BaseSD
"""
copy from
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl_img2img.py
"""
class Refiner(nn.Module, BaseSD):
    """
    SDXL Refiner to clean the artifact
    """
    @dataclass
    class Config(BaseModule.Config):
        num_inference_steps: int = 100
        height: int = 1024 # keep it as it is otherwise poor quality
        width: int = 1024
        infer_batch_size: int = 1
        negative_prompt: Optional[str] = None

        pretrained_refiner_name_or_path: str = "stabilityai/stable-diffusion-xl-refiner-1.0"
        weights_dtype: str = torch.float16
        device: str = "cuda"

    def __init__(self, **kwargs):  
        super().__init__()
        self.cfg = self.Config()
        default_args = {
            'pretrained_refiner_name_or_path': self.cfg.pretrained_refiner_name_or_path,
            'weights_dtype': self.cfg.weights_dtype,
            'device': torch.device("cuda")
        }

        for key, value in {**default_args, **kwargs}.items():
            setattr(self.cfg, key, value)

        self.device = self.cfg.device
        self.pipe = DiffusionPipeline.from_pretrained(
                self.cfg.pretrained_refiner_name_or_path,
                torch_dtype=self.cfg.weights_dtype,
        ).to(self.device)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        masks: torch.FloatTensor = None,
        bg_latents: torch.FloatTensor = None,
        strength: float = 0.3,
        num_inference_steps: int = 50,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        clip_skip: Optional[int] = None,
    ):
       
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self.device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Preprocess image
        image = self.pipe.image_processor.preprocess(image)

        # 5. Prepare timesteps
        def denoising_value_valid(dnv):
            return isinstance(denoising_end, float) and 0 < dnv < 1

        self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.pipe.get_timesteps(
            num_inference_steps, strength, device, denoising_start=denoising_start if denoising_value_valid else None
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        add_noise = True if denoising_start is None else False
        # 6. Prepare latent variables
        latents = self.pipe.prepare_latents(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
            add_noise,
        )
        # 7. Prepare extra step kwargs.
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)

        height, width = latents.shape[-2:]
        height = height * self.pipe.vae_scale_factor
        width = width * self.pipe.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 8. Prepare added time ids & embeddings
        if negative_original_size is None:
            negative_original_size = original_size
        if negative_target_size is None:
            negative_target_size = target_size

        add_text_embeds = pooled_prompt_embeds
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_time_ids, add_neg_time_ids = self.pipe._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            aesthetic_score,
            negative_aesthetic_score,
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim
        )
        add_time_ids = add_time_ids.repeat(batch_size * num_images_per_prompt, 1)

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_neg_time_ids = add_neg_time_ids.repeat(batch_size * num_images_per_prompt, 1)
            add_time_ids = torch.cat([add_neg_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device)

        # 9. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 9.1 Apply denoising_end
        if (
            denoising_end is not None
            and denoising_start is not None
            and denoising_value_valid(denoising_end)
            and denoising_value_valid(denoising_start)
            and denoising_start >= denoising_end
        ):
            raise ValueError(
                f"`denoising_start`: {denoising_start} cannot be larger than or equal to `denoising_end`: "
                + f" {denoising_end} when using type float."
            )
        elif denoising_end is not None and denoising_value_valid(denoising_end):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            if i < timesteps.shape[0] - 1 and bg_latents is not None:
                noise = randn_tensor(
                    bg_latents.shape, 
                    generator=generator, 
                    device=self.device, 
                    dtype=bg_latents.dtype,
                )
                noisy_bg_latents = self.scheduler.add_noise(bg_latents, noise, timesteps[i+1][None])
                latents = masks * latents + (1 - masks) * noisy_bg_latents
          
        return latents
        