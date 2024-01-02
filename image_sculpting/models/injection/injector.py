from diffusers import DDIMScheduler

from threestudio.utils.typing import *
import image_sculpting.models.injection as inj 

class Injector:
    """
    The Injector class handles the injection process for SDXL
    """
    def configure_injector(
        self,
        pipe,
        num_inference_steps: int,
        self_injection_t: float,
        conv_injection_t: float,
        up_blocks_dict: Dict, 
        res_blocks_dict: Dict,
        device,
    ) -> None:
        """
        Setting up the injector.

        Args:
            pipe: SD pipeline
            num_inference_steps (int): 
                Total number of inference steps in the pipeline.
            self_injection_t (float): 
                The timestep for self-injection in the pipeline.
            conv_injection_t (float): 
                The timestep for convolutional injection in the pipeline.
            
            up_blocks_dict (Dict): 
                Define which attention layers to inject
            res_blocks_dict (Dict): 
                Define which resnet layers to inject
        """
        self.pipe = pipe 
        del pipe.scheduler
        pipe.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.pretrained_model_name_or_path, 
            subfolder="scheduler"
        )
        
        self.device = device
        self.construct_module_paths(up_blocks_dict, res_blocks_dict)
        for key in self.MODULES_DICT:
            self.MODULES_DICT[key] = inj.find(key).forward
       
       
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)

        injection_times = {
            "self": max(0, int(self_injection_t * num_inference_steps)),
            "conv": max(0, int(conv_injection_t * num_inference_steps))
        }

        for inj_type, time in injection_times.items():
            if time <= 0:
                setattr(self, f"{inj_type}_injection_timesteps", [])
            else:
                setattr(self, f"{inj_type}_injection_timesteps", pipe.scheduler.timesteps[:time])
        self.original_forwards = {}
       
    def overwrite_forward_methods(self):
        """
        Overwrites the forward pass for injection.
        """
        for module_name, forward in self.PATHS.items():
            self._register_module(module_name, forward)
    
    def reset_forward_methods(self):
        """
        Reset forward pass to that of SD
        """
        for name, original_forward in self.original_forwards.items():
            module = self._get_module(tuple(name.split('.')))
            module.forward = original_forward
           
    def construct_module_paths(
        self,
        up_blocks_dict,
        res_blocks_dict,
    ):
        """
        Constructs paths to the modules within the network where injections will be applied.

        Args:
            up_blocks_dict (Dict): Dictionary specifying the self-attention blocks.
            res_blocks_dict (Dict): Dictionary specifying the residual blocks.
        """
        self.PATHS = {}

      
        name_template = ["unet", "up_blocks", -1, 
                        "attentions", -1, "transformer_blocks", 
                        -1]
        
        for up_block_key, up_block_values in up_blocks_dict.items():
            for attention_key, _ in up_block_values.items():
                for transformer_block_key in up_block_values[attention_key]:

                    path_key = name_template.copy()

                    path_key[2] = up_block_key
                    path_key[4] = attention_key
                    path_key[6] = transformer_block_key

                    self.PATHS[tuple(path_key + ["attn1"])] = "AttnProcessor"
                   
        
        res_template = ["unet", "up_blocks", -1, "resnets", -1]
        for up_block_key, resnets_range in res_blocks_dict.items():
            for resnet_key in resnets_range:
                path_key = res_template.copy()

                path_key[2] = up_block_key
                path_key[4] = resnet_key

                self.PATHS[tuple(path_key)] = "ResnetBlock2D"
                
        self.MODULES_DICT = {
            "AttnProcessor": None,
            "ResnetBlock2D": None,
        }

    def _register_time(self, t):
        """
        Registers time for each block
        """
        for module_name in self.PATHS:
            module = self._get_module(module_name)
            setattr(module, "t", t)
    
    def _register_module(
        self, 
        module_name, 
        forward_name,
    ):
        """
        Registers injection schedule and forward pass
        """
        module = self._get_module(module_name)
        forward = self.MODULES_DICT.get(forward_name)
        name = '.'.join(map(str, module_name))
        if forward:
            self.original_forwards[name] = module.forward
            module.forward = forward(module)
        else:
            raise AttributeError("No forward method found in the provided module information.")
        attributes_to_set = {
            "conv_injection_timesteps": self.conv_injection_timesteps,
            "qk_injection_timesteps": self.self_injection_timesteps,
            "module_name": '.'.join(str(cha) for cha in module_name),
        }
        self._set_attributes_for_module(module, attributes_to_set)
        
    def _get_module(self, module_name):
        module = self.pipe
        for path in module_name:
            module = module[path] if isinstance(path, int) else getattr(module, path)

        return module

    def _set_attributes_for_module(self, module, attributes_to_set):
        for attr_name, attr_value in attributes_to_set.items():
            setattr(module, attr_name, attr_value)