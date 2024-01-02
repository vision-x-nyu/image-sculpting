from typing import Any, Dict, Optional
from math import sqrt
import numpy as np 

import torch
import torch.nn.functional as F
from torchvision import transforms
from einops import rearrange
from sklearn.decomposition import PCA

_EPS = 1e-12
"""
Copy some part from 
    https://github.com/MichalGeyer/pnp-diffusers
"""
np_to_torch = transforms.Compose([transforms.ToTensor()])

class Visualizer:
    @staticmethod
    def visualize(feature_maps):
        pca_result = PCA(n_components=3).fit_transform(feature_maps)
        dim = int(sqrt(pca_result.shape[0]))
        feature_maps_pca = pca_result.reshape(dim, dim, 3)
        min_vals = feature_maps_pca.min(axis=(0, 1))
        max_vals = feature_maps_pca.max(axis=(0, 1))
        normalized_maps = (feature_maps_pca - min_vals) / (max_vals - min_vals + _EPS)
        torch_maps = np_to_torch(np.clip(normalized_maps, 0., 1.0))
        return torch_maps

    @staticmethod
    def process_and_visualize_attn(self_attn_map):
        if len(self_attn_map.shape) != 3:
            raise "Not self attention map. Shape should be [b, n^2, n^2]"
        mid_index = int(self_attn_map.shape[0] / 2)
        feature_maps = rearrange(self_attn_map[mid_index:], 'h n m -> n (h m)').detach().cpu().numpy()
        return Visualizer.visualize(feature_maps)

    @staticmethod
    def process_and_visualize_feature(tensor):
        if len(tensor.shape) != 4:
            raise "Not feature map. Shape should be [b, c, h, w]"
        reshaped_tensor = tensor[int(tensor.shape[0] / 2):]
        reshaped_tensor = rearrange(reshaped_tensor, 'b c h w -> (h w) (b c)').detach().cpu().numpy()
        return Visualizer.visualize(reshaped_tensor)