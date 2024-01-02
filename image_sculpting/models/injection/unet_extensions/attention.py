from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import torch
import torch.nn.functional as F

import image_sculpting.models.injection as inj 
from image_sculpting.models.injection.unet_extensions.visualizer import Visualizer 
"""
Copy some part from 
    https://github.com/MichalGeyer/pnp-diffusers
"""
@inj.register("AttnProcessor")
class AttnProcessor:
    @staticmethod
    def forward(module):
        to_out = module.to_out
        if isinstance(to_out, torch.nn.modules.container.ModuleList):
            to_out = module.to_out[0]
        else:
            to_out = module.to_out
        def _forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
        ):
            batch_size, sequence_length, dim = hidden_states.shape
            h = module.heads

            is_cross = encoder_hidden_states is not None
            
            encoder_hidden_states = encoder_hidden_states if is_cross else hidden_states
            
            q = module.to_q(hidden_states)
            k = module.to_k(encoder_hidden_states)
            
            # inject
            if hasattr(module, "qk_injection_timesteps") \
                and module.t in module.qk_injection_timesteps:
                source_batch_size = int(hidden_states.shape[0] // 3)

                q[source_batch_size:] = torch.cat([q[:source_batch_size]] * 2)
                k[source_batch_size:] = torch.cat([k[:source_batch_size]] * 2)
                
            q = module.head_to_batch_dim(q)
            k = module.head_to_batch_dim(k)

            v = module.to_v(encoder_hidden_states)
            v = module.head_to_batch_dim(v)
            
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * module.scale
            # attn [batch, h, w]
            self_attn_map = sim.softmax(dim=-1)
            
            if attention_mask is not None:
                attention_mask = attention_mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                attention_mask = attention_mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~attention_mask, max_neg_value)

            out = torch.einsum("b i j, b j d -> b i d", self_attn_map, v)
            out = module.batch_to_head_dim(out)
            out = to_out(out)
            return out

        return _forward
       