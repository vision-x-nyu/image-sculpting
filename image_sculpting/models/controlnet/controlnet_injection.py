from dataclasses import dataclass, field


from threestudio.utils.typing import *
import numpy as np
from skimage import io
import cv2 
import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLPipeline,
    ControlNetModel,
    DiffusionPipeline,
)
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.utils.torch_utils import randn_tensor

from threestudio.utils.base import BaseObject, BaseModule
import image_sculpting
import image_sculpting.models.injection as inj 
from image_sculpting.models.injection.injector import Injector 
from image_sculpting.models.utils.base import BaseSD
from image_sculpting.models.utils.prompt_processors import PromptProcessor
from image_sculpting.models.utils.refiner import Refiner

"""
Copy some part from
    https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/controlnet/pipeline_controlnet_sd_xl.py
"""
@image_sculpting.register("ours")
class Ours(BaseModule, Injector, BaseSD):
    """
    Coarse-to-fine generative enhancement.
    """
    @dataclass
    class Config(BaseModule.Config):
        """
        Attributes:
            - pretrained_model_name_or_path (str): Pretrained model SDXL.
            - pretrained_controlnet (str): Pretrained controlnet model.
            - seed (int): Seed for random number.
            - num_inference_steps (int): Number of inference steps.
            - infer_batch_size (int): Batch size for inference.
            - half_precision_weights (bool): Flag to use half-precision weights.
            - height (int): Height of the output image.
            - width (int): Width of the output image.
            - guidance_scale (float): CFG scale.
            - negative_prompt (Optional[str]): Negative prompt.
            - instance_prompt (str): text prompt
            - inversion_prompt (str): text prompt used in DDIM Inversion

            - lora_weights: (str): path to LoRA
            - controlnet_conditioning_scale: 
                condition scale as recommended here https://huggingface.co/docs/diffusers/using-diffusers/controlnet
            - use_masks (bool): Flag to use mask in background
            - kernel_size (int): dilation size for mask

            - use_ddim_init (bool): Flag to use inversion noise as a starting point

            - self_injection_t (float): when to stop self-attention injection from T = 0 to T = 1
            - conv_injection_t (float): when to stop feature injection from T = 0 to T = 1
            - up_blocks_dict (Dict): Dictionary to specify which self-attention layers to inject
            - res_blocks_dict (Dict): Dictionary to specify which resnet layers to inject


            - use_refiner (bool): Flag to use SDXL Refiner
            - pretrained_refiner_name_or_path (str): Pretrained model SDXL Refiner.
            - high_noise_frac (float): Timesteps to start using refiner
        """
        pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-xl-base-1.0"
        pretrained_controlnet: str = "diffusers/controlnet-depth-sdxl-1.0"
        seed: int = 1
        num_inference_steps: int = 100
        infer_batch_size: int = 1
        half_precision_weights: bool = True
        height: int = 1024
        width: int = 1024
        guidance_scale: float = 5.0
        negative_prompt: Optional[str] = "ugly, blurry, low res, unrealistic, worst quality, low quality, jpeg artifacts"
 
        instance_prompt: str = ""
        inversion_prompt: str = ""

        lora_weights: str = "./runs/ours/horse"
        controlnet_conditioning_scale: float = 0.5
        use_masks: bool = True
        kernel_size: int = 41

        use_ddim_init: bool = True

        self_injection_t: float = 0.5
        conv_injection_t: float = 0.8
        up_blocks_dict: Dict[int, Dict[int, List[int]]] = field(default_factory=lambda: {
            0: {0: list(range(10)), 1: list(range(10)), 2: list(range(10))},
            1: {0: list(range(2)), 1: list(range(2)), 2: list(range(2))}
        })
        res_blocks_dict: Dict[int, List[int]] = field(default_factory=lambda: {
            0: list(range(3)) 
        })

        #refiner
        use_refiner: bool = True 
        pretrained_refiner_name_or_path: str = "stabilityai/stable-diffusion-xl-refiner-1.0"
        high_noise_frac: float = 0.9
     
    cfg: Config 
    def configure(self):
        """
        Configures the pipeline
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
        control_net = ControlNetModel.from_pretrained(
            self.cfg.pretrained_controlnet, 
            torch_dtype=self.weights_dtype
        )
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            controlnet=control_net,
            torch_dtype=self.weights_dtype, 
        ).to(self.device)
        self.pipe.load_lora_weights(self.cfg.lora_weights)
        #self.pipe.enable_xformers_memory_efficient_attention()
       
        # self.pipe.enable_model_cpu_offload()
        # self.pipe.enable_vae_tiling()
        self.kernel = np.ones((self.cfg.kernel_size, self.cfg.kernel_size), np.uint8)
       
        self.configure_injector(
            pipe=self.pipe,
            num_inference_steps=self.cfg.num_inference_steps,
            self_injection_t=self.cfg.self_injection_t,
            conv_injection_t=self.cfg.conv_injection_t,
            up_blocks_dict=self.cfg.up_blocks_dict,
            res_blocks_dict=self.cfg.res_blocks_dict,
            device=self.device,
        )
        self.prompt_processors = PromptProcessor()

        if self.cfg.use_refiner:
            self.refiner = Refiner(
                pretrained_refiner_name_or_path=self.cfg.pretrained_refiner_name_or_path,
                weights_dtype=self.weights_dtype,
                device=self.device
            )

           
    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def _sample(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: Optional[torch.FloatTensor] = None,
        latents_inv: Optional[Float[Tensor, "T B 4 H W"]] = None,
        masks: Optional[Float[Tensor, "B 1 H W"]] = None,
        background_img: Optional[Float[Tensor, "B 3 H W"]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        mesh_azimuth: Optional[Float[Tensor, "B"]] = None,
        num_inference_steps: int = 50,
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
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Generates an enhanced image.

        Args:
            prompt (Union[str, List[str]]): Text prompt(s) for image generation.
            image (torch.FloatTensor): Depth map.
            latents_inv (Float[Tensor, "T B 4 H W"]): Inverted latents for.
            masks (Float[Tensor, "B 1 H W"]): Optional masks for localized image generation.
            background_img (Float[Tensor, "B 3 H W"]): Background image tensor.
            mesh_azimuth: angle when rotating the mesh
        Returns:
            torch.Tensor: Generated image tensor.
        """
        #controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet
        controlnet = self.controlnet
        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]

    
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

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
    
       
        prompt, negative_prompt = self.prompt_processors(prompt, negative_prompt, mesh_azimuth)
       
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
        
        # 4. Prepare image
       
        if isinstance(controlnet, ControlNetModel):
            image = self.pipe.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            images = []

            for image_ in image:
                image_ = self.pipe.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            image = images
            
            height, width = image[0].shape[-2:]
        else:
            assert False
        image = torch.cat([
            image[1:],
            image
        ], 0)

        # 5. Prepare timesteps
        
        timesteps = self.scheduler.timesteps
        if self.cfg.use_ddim_init:
            latents = latents_inv[-1]
        else:
            latents = self.pipe.prepare_latents(
                    batch_size * num_images_per_prompt,
                    self.unet.config.in_channels,
                    height,
                    width,
                    prompt_embeds.dtype,
                    device,
                    generator,
                    latents,
            )

        # 6. Prepare latent variables
        bg_latents = None 
        masks_reshaped = None 
        if self.cfg.use_masks:
            masks_reshaped = F.interpolate(
                masks,
                (latents.shape[2], latents.shape[3]),
                mode="nearest",
            )
            background_img = F.interpolate(
                background_img,
                (self.cfg.height, self.cfg.width), 
                mode="bilinear"
            )
           
            bg_latents = self._encode_images(self.vae, background_img, self.weights_dtype)
            
        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 7.2 Prepare added time ids & embeddings
        if isinstance(image, list):
            original_size = original_size or image[0].shape[-2:]
        else:
            original_size = original_size or image.shape[-2:]
        target_size = target_size or (height, width)
        if self.text_encoder_2 is None:
            text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.text_encoder_2.config.projection_dim

        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self.pipe._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self.pipe._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
            )
        else:
            negative_add_time_ids = add_time_ids

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
        (
            inv_prompt_embeds,
            _,
            inv_pooled_prompt_embeds,
            _
        ) = self.pipe.encode_prompt(
            prompt=self.cfg.inversion_prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=False,
        )
       
        prompt_embeds = torch.cat([inv_prompt_embeds, prompt_embeds], 0)
        add_text_embeds = torch.cat([inv_pooled_prompt_embeds, add_text_embeds], 0)
        add_time_ids = torch.cat([add_time_ids[1:], add_time_ids], 0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        if denoising_end is not None and isinstance(denoising_end, float) and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        for i, t in enumerate(timesteps):
            self._register_time(t)
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            geometry_latents = latents_inv[-i-1]
            latent_model_input = torch.cat([geometry_latents, latent_model_input], 0)
            
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

            # controlnet(s) inference
            if guess_mode and do_classifier_free_guidance:
                # Infer ControlNet only for the conditional batch.
                control_model_input = latents
                control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                controlnet_added_cond_kwargs = {
                    "text_embeds": add_text_embeds.chunk(2)[1],
                    "time_ids": add_time_ids.chunk(2)[1],
                }
            else:
                control_model_input = latent_model_input
                controlnet_prompt_embeds = prompt_embeds
                controlnet_added_cond_kwargs = added_cond_kwargs

            if isinstance(controlnet_keep[i], list):
                cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
            else:

                controlnet_cond_scale = controlnet_conditioning_scale
                if isinstance(controlnet_cond_scale, list):
                    controlnet_cond_scale = controlnet_cond_scale[0]
                cond_scale = controlnet_cond_scale * controlnet_keep[i]
           
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                control_model_input,
                t,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=image,
                conditioning_scale=cond_scale,
                guess_mode=guess_mode,
                added_cond_kwargs=controlnet_added_cond_kwargs,
                return_dict=False,
            )

            if guess_mode and do_classifier_free_guidance:
                # Infered ControlNet only for the conditional batch.
                # To apply the output of ControlNet to both the unconditional and conditional batches,
                # add 0 to the unconditional batch to keep it unchanged.
                down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

            # predict the noise residual
            
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            
            # perform guidance

            if do_classifier_free_guidance:
                _, noise_pred_uncond, noise_pred_text = noise_pred.chunk(3)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            
            if self.cfg.use_masks and i < timesteps.shape[0] - 1:
                noise = randn_tensor(
                    bg_latents.shape, 
                    generator=generator, 
                    device=self.device, 
                    dtype=bg_latents.dtype,
                )
                noisy_bg_latents = self.scheduler.add_noise(bg_latents, noise, timesteps[i+1][None])
                latents = masks_reshaped * latents + (1 - masks_reshaped) * noisy_bg_latents
         
        no_rare_token_prompt = prompt.replace("sks ", "")
        if self.cfg.use_refiner:
            latents = self.refiner(
                    prompt=no_rare_token_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    denoising_start=denoising_end,
                    image=latents,
                    bg_latents=bg_latents,
                    masks=masks_reshaped,
                    generator=generator,
                    output_type="latent",
                )
        # manually for max memory savings
        
        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.pipe.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            if needs_upcasting:
                self.pipe.upcast_vae()
                latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        
        image = self._decode_latents(self.vae, latents)
        return image
    
    def __call__(
        self,
        depth: Float[Tensor, "B 1 H W"],
        latents_inv: Float[Tensor, "T B 4 H W"],
        masks: Optional[Float[Tensor, "B 1 H W"]] = None,
        background_img: Optional[Float[Tensor, "B 3 H W"]] = None,
        mesh_azimuth: Optional[Float[Tensor, "B"]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Callable method to output high-detailed image
        Args:
            depth (Float[Tensor, "B 1 H W"]): Depth map.
            latents_inv (Float[Tensor, "T B 4 H W"]): Inverted latents from inversion.
            masks (Optional[Float[Tensor, "B 1 H W"]]): Optional masks for localized generation.
            background_img (Optional[Float[Tensor, "B 3 H W"]]): Background image tensor.
            mesh_azimuth (Optional[Float[Tensor, "B"]]): Mesh azimuth from rotation.
        Returns:
            torch.Tensor: The generated image.
        """
        if self.cfg.use_masks:
            masks_np = masks[0, 0].detach().cpu().numpy()
            masks = cv2.dilate(masks_np, self.kernel, iterations=1)
            
            masks = torch.from_numpy(masks[None, None]).to(self.device).to(self.weights_dtype)
          
        img = self._sample(
            prompt=self.cfg.instance_prompt,
            image=depth,
            latents_inv=latents_inv.to(self.weights_dtype),
            masks=masks,
            background_img=background_img.to(self.weights_dtype),
            mesh_azimuth=mesh_azimuth,
            num_inference_steps=self.cfg.num_inference_steps,
            negative_prompt=self.cfg.negative_prompt,
            controlnet_conditioning_scale=self.cfg.controlnet_conditioning_scale,
            generator=torch.Generator(device=self.device).manual_seed(self.cfg.seed),
            denoising_end=self.cfg.high_noise_frac if self.cfg.use_refiner else None,
        )
        
        return img 