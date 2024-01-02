from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch
import torch.nn.functional as F

import image_sculpting.models.injection as inj 
from image_sculpting.models.injection.unet_extensions.visualizer import Visualizer 
"""
Copy some part from 
    https://github.com/MichalGeyer/pnp-diffusers
"""
@inj.register("ResnetBlock2D")
class ResnetBlock2D:
    @staticmethod
    def forward(module):
        def _forward(
            input_tensor, 
            temb, 
            scale: float = 1.0,
        ):
            hidden_states = input_tensor

            if module.time_embedding_norm == "ada_group" or module.time_embedding_norm == "spatial":
                hidden_states = module.norm1(hidden_states, temb)
            else:
                hidden_states = module.norm1(hidden_states)

            hidden_states = module.nonlinearity(hidden_states)

            if module.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = (
                    module.upsample(input_tensor, scale=scale)
                    if isinstance(module.upsample, Upsample2D)
                    else module.upsample(input_tensor)
                )
                hidden_states = (
                    module.upsample(hidden_states, scale=scale)
                    if isinstance(module.upsample, Upsample2D)
                    else module.upsample(hidden_states)
                )
            elif module.downsample is not None:
                input_tensor = (
                    module.downsample(input_tensor, scale=scale)
                    if isinstance(module.downsample, Downsample2D)
                    else module.downsample(input_tensor)
                )
                hidden_states = (
                    module.downsample(hidden_states, scale=scale)
                    if isinstance(module.downsample, Downsample2D)
                    else module.downsample(hidden_states)
                )

            #hidden_states = module.conv1(hidden_states, scale)
            hidden_states = module.conv1(hidden_states)

            if module.time_emb_proj is not None:
                if not module.skip_time_act:
                    temb = module.nonlinearity(temb)
                #temb = module.time_emb_proj(temb, scale)[:, :, None, None]
                temb = module.time_emb_proj(temb)[:, :, None, None]

            if temb is not None and module.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            if module.time_embedding_norm == "ada_group" or module.time_embedding_norm == "spatial":
                hidden_states = module.norm2(hidden_states, temb)
            else:
                hidden_states = module.norm2(hidden_states)

            if temb is not None and module.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = module.nonlinearity(hidden_states)

            hidden_states = module.dropout(hidden_states)
            #hidden_states = module.conv2(hidden_states, scale)
            hidden_states = module.conv2(hidden_states)
            # inject
            if hasattr(module, "conv_injection_timesteps") \
                and module.t in module.conv_injection_timesteps:
                source_batch_size = int(hidden_states.shape[0] // 3)
                hidden_states[source_batch_size:] = torch.cat([hidden_states[:source_batch_size]] * 2)
            if module.conv_shortcut is not None:
                #input_tensor = module.conv_shortcut(input_tensor, scale)
                input_tensor = module.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / module.output_scale_factor

            return output_tensor

        return _forward