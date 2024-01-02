from dataclasses import dataclass
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torchvision import transforms
import numpy as np
import os
import trimesh
import pyrender
import json 
from omegaconf import OmegaConf
import image_sculpting
from threestudio.utils.base import BaseModule

from threestudio.utils.typing import *
os.environ['PYOPENGL_PLATFORM'] = 'egl'
_EPS = 1e-12
"""
Copy from DeepView
    https://augmentedperception.github.io/deepview/
"""

rainbow = [
  [0.18995, 0.07176, 0.23217], [0.22500, 0.16354, 0.45096],
  [0.25107, 0.25237, 0.63374], [0.26816, 0.33825, 0.78050],
  [0.27628, 0.42118, 0.89123], [0.27543, 0.50115, 0.96594],
  [0.25862, 0.57958, 0.99876], [0.21382, 0.65886, 0.97959],
  [0.15844, 0.73551, 0.92305], [0.11167, 0.80569, 0.84525],
  [0.09267, 0.86554, 0.76230], [0.12014, 0.91193, 0.68660],
  [0.19659, 0.94901, 0.59466], [0.30513, 0.97697, 0.48987],
  [0.42778, 0.99419, 0.38575], [0.54658, 0.99907, 0.29581],
  [0.64362, 0.98999, 0.23356], [0.72596, 0.96470, 0.20640],
  [0.80473, 0.92452, 0.20459], [0.87530, 0.87267, 0.21555],
  [0.93301, 0.81236, 0.22667], [0.97323, 0.74682, 0.22536],
  [0.99314, 0.67408, 0.20348], [0.99593, 0.58703, 0.16899],
  [0.98360, 0.49291, 0.12849], [0.95801, 0.39958, 0.08831],
  [0.92105, 0.31489, 0.05475], [0.87422, 0.24526, 0.03297],
  [0.81608, 0.18462, 0.01809], [0.74617, 0.13098, 0.00851],
  [0.66449, 0.08436, 0.00424], [0.47960, 0.01583, 0.01055]
]

magma = [
  [0.001462, 0.000466, 0.013866], [0.013708, 0.011771, 0.068667],
  [0.039608, 0.031090, 0.133515], [0.074257, 0.052017, 0.202660],
  [0.113094, 0.065492, 0.276784], [0.159018, 0.068354, 0.352688],
  [0.211718, 0.061992, 0.418647], [0.265447, 0.060237, 0.461840],
  [0.316654, 0.071690, 0.485380], [0.366012, 0.090314, 0.497960],
  [0.414709, 0.110431, 0.504662], [0.463508, 0.129893, 0.507652],
  [0.512831, 0.148179, 0.507648], [0.562866, 0.165368, 0.504692],
  [0.613617, 0.181811, 0.498536], [0.664915, 0.198075, 0.488836],
  [0.716387, 0.214982, 0.475290], [0.767398, 0.233705, 0.457755], 
  [0.816914, 0.255895, 0.436461], [0.863320, 0.283729, 0.412403],
  [0.904281, 0.319610, 0.388137], [0.937221, 0.364929, 0.368567],
  [0.960949, 0.418323, 0.359630], [0.976690, 0.476226, 0.364466],
  [0.986700, 0.535582, 0.382210], [0.992785, 0.594891, 0.410283],
  [0.996096, 0.653659, 0.446213], [0.997325, 0.711848, 0.488154],
  [0.996898, 0.769591, 0.534892], [0.995131, 0.827052, 0.585701],
  [0.992440, 0.884330, 0.640099], [0.987053, 0.991438, 0.749504]
]

def colorLerp(a, b, p):
    q = 1 - p
    return [q * a[0] + p * b[0], q * a[1] + p * b[1], q * a[2] + p * b[2]]

def colorHeat(f, min_val, max_val, colors):
    f = (f - min_val) / (max_val - min_val)
    f = f * (len(colors) - 1)
    idx = int(f)
    if idx >= len(colors) - 1:
        return colors[-1]
    elif idx < 0:
        return colors[0]
    else:
        frac = f - idx
        return colorLerp(colors[idx], colors[idx + 1], frac)

@image_sculpting.register("mesh-renderer")
class MeshRenderer(BaseModule):
    
    BASE_COLOR_FACTOR = np.array([1, 1, 1, 1])
    AMBIENT_LIGHT = np.array([1., 1., 1., 1.])

    @dataclass
    class Config(BaseModule.Config):
        """
        Attributes:
            deformed_mesh (str): File path to the deformed mesh
            original_mesh (str): File path to the original mesh
            render_height (int): Height of the rendered image.
            render_width (int): Width of the rendered image.
            height (int): Height of the output image.
            width (int): Width of the output image.
            inserted_object (str): File path to the extra 3D mesh

            ambient_light_color (Tuple[float, float, float]): values for ambient light in the scene.
            diffuse_light_color (Tuple[float, float, float]): อalues for diffuse light in the scene.
            
            intensity (float): Intensity of the lighting in the scene.
            log_file (str): File path to the JSON file containing information about 
                            image preprocessing (scaling and moving image back to its original location).
            img_name (str): a path for the input image
        """
        deformed_mesh: str = "???.obj"
        original_mesh: str = "???.obj"

        render_height: int = 2048
        render_width: int = 2048
        height: int = 1024
        width: int = 1024

        ambient_light_color: Tuple[float, float, float] = (0.1, 0.1, 0.1)
        diffuse_light_color: Tuple[float, float, float] = (0.9, 0.9, 0.9)

        intensity: float = 300.0
        log_file: str = "./sculpting_data/data.json"
        img_name: str = ""

        inserted_object: Optional[str] = None 

    cfg: Config
    def configure(
        self,
        ) -> None:
        parts = self.cfg.deformed_mesh.split(os.path.sep)
        """
        Configure renderer
        """
       
        self.r = pyrender.OffscreenRenderer(self.cfg.render_width, self.cfg.render_height)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pil2tensor = transforms.ToTensor()
        with open(self.cfg.log_file, 'r') as json_file:
            self.log = json.load(json_file)
        if self.cfg.img_name == "":
            self.cfg.img_name = '/'.join(self.cfg.original_mesh.split('/')[-4:-2])
        self.transformation_data = self.log[self.cfg.img_name]
       
        if self.cfg.inserted_object:
            if hasattr(self.cfg.inserted_object, '_content'):
                inserted_objects = OmegaConf.to_container(self.cfg.inserted_object)
            else:
                inserted_objects = self.cfg.inserted_object

            if isinstance(inserted_objects, list):
                self.inserted_object_mesh = [trimesh.load(mesh, force='mesh') for mesh in inserted_objects]
            else:
                self.inserted_object_mesh = trimesh.load(inserted_objects, force='mesh')
        else:
            self.inserted_object_mesh = None
        self.original_location = None 
        self.original_mesh = trimesh.load(self.cfg.original_mesh, process=False)

    def forward(
        self,
        intrinsics: Float[Tensor, "B 4 4"],
        c2w: Float[Tensor, "B 3 3"],
        mesh_azimuth: Float[Tensor, "B"],
        **kwargs,
    ):
        """
        Render mesh.

        Args:
            intrinsics (Float[Tensor, "B 4 4"]): The camera intrinsic parameters.
            c2w (Float[Tensor, "B 3 3"]): The camera-to-world transformation matrix.
            mesh_azimuth (Float[Tensor, "B"]): The azimuth angle for mesh rotation.

        Returns:
            the color, depth, and depth RGB images in Dictionary
        """
        deformed_mesh = trimesh.load(self.cfg.deformed_mesh, process=False)
        com = deformed_mesh.centroid

        rotation = self.y_rotation_matrix(np.deg2rad(mesh_azimuth[0].item())) 

        deformed_mesh.apply_translation(-com)
        deformed_mesh.apply_transform(rotation)
        deformed_mesh.apply_translation(com)
       
        intrinsics = intrinsics[0]
        c2w = c2w[0]
        camera = pyrender.camera.IntrinsicsCamera(
            fx=intrinsics[0, 0].item(),
            fy=intrinsics[1, 1].item(),
            cx=self.cfg.render_height / 2,
            cy=self.cfg.render_width / 2,
        )
        
        pose = c2w.to('cpu').detach().numpy().copy()
       
        if self.original_location is None:
            # render original mesh in order to get the original location for 
            # moving deformed mesh back 
            rgba_array, _, _ = self._render_helper(
                self.original_mesh, 
                pose, 
                camera, 
                original_mesh=True,
                **kwargs
            )
            rmin, rmax, cmin, cmax = self._find_bounding_box(rgba_array)
            self.original_location = (rmin, cmin)
            scale_factor = 1 / self.transformation_data["scale_factor"]
            scaled_size = (
                int((rmax - rmin + 1) * scale_factor),
                int((cmax - cmin + 1) * scale_factor), 
            )
            self.start_position = ((self.cfg.height - scaled_size[0]) // 2,
                                    (self.cfg.width - scaled_size[1]) // 2)

        color, depth, depth_rgb = self._render_helper(
            deformed_mesh, 
            pose, 
            camera, 
            **kwargs
        )
        
        rmin, rmax, cmin, cmax = self._find_bounding_box(color)
        
        color = self.pil2tensor(self._recover_original_image(color, rmin, rmax, cmin, cmax))[None].float().to(self.device)
        depth = self._recover_original_image(depth, rmin, rmax, cmin, cmax).convert("RGB")
       
        depth_rgb = self._recover_original_image(depth_rgb, rmin, rmax, cmin, cmax)
      
        return {
            "color": color,
            "depth": depth,
            "depth_rgb": depth_rgb,
        }
    
    def _render_helper(self,
            mesh: pyrender.mesh.Mesh,
            pose: np.array,
            camera: pyrender.camera.IntrinsicsCamera,
            ambient_ratio: Optional[float] = None,
            shading: Optional[str] = None,
            original_mesh: bool =False,
            **kwargs,
        ) -> np.array:
        """
        Render the mesh using the provided camera settings and pose.
        """
        scene = pyrender.Scene(ambient_light=np.array(self.cfg.ambient_light_color))
        scene.add(camera, pose=pose)
        

        self._add_mesh_to_scene(scene, mesh)
        #scene.add(mesh)
        if not original_mesh and self.inserted_object_mesh:
            if isinstance(self.inserted_object_mesh, list):
                for m in self.inserted_object_mesh:
                    self._add_mesh_to_scene(scene, m)
            else:
                self._add_mesh_to_scene(scene, self.inserted_object_mesh)
     
        light = pyrender.PointLight(color=np.array(self.cfg.diffuse_light_color), 
                                    intensity=self.cfg.intensity)
        light_pose = np.copy(pose)
       
        scene.add(light, pose=light_pose)
        color, depth = self.r.render(scene, 
                                 flags=pyrender.constants.RenderFlags.RGBA)

        depth[depth == 0] = np.inf
        finite_depth_mask = np.isfinite(depth)
        
        depth = 1 / (depth + _EPS) 
        finite_depth_min = depth[finite_depth_mask].min()
        finite_depth_max = depth[finite_depth_mask].max()
        
      
        normalized_depth = (depth - finite_depth_min) / (finite_depth_max - finite_depth_min)
        depth = np.repeat(normalized_depth[..., None], 3, axis=2)
        depth = (depth * 255.0).clip(0, 255).astype(np.uint8)
        
        depth_rgb = self.depth_to_color(normalized_depth, finite_depth_mask)
        depth_rgb = (depth_rgb * 255.0).clip(0, 255).astype(np.uint8)
    
        
        alpha_channel = (finite_depth_mask * 255).astype(np.uint8)
        color_writable = color.copy()
        color_writable[..., 3] = alpha_channel
      
        return color_writable, depth, depth_rgb
    
    def depth_to_color(
        self, 
        normalized_depth: np.array, 
        depth_masked: np.array, 
        colormap=rainbow
    ) -> np.array:
        """
        Convert normalized depth values to color based on a colormap.
        """

        scaled_depth = normalized_depth * (len(colormap) - 1)
        idx = np.floor(scaled_depth).astype(int)
        frac = scaled_depth - idx

        idx_clipped = np.clip(idx, 0, len(colormap) - 2)
        next_idx_clipped = np.clip(idx + 1, 0, len(colormap) - 1)

        color_start = np.array(colormap)[idx_clipped]
        color_end = np.array(colormap)[next_idx_clipped]
        depth_colored = (1 - frac[..., np.newaxis]) * color_start + frac[..., np.newaxis] * color_end

        depth_colored[~depth_masked] = [0, 0, 0]  
        return depth_colored
    
    def y_rotation_matrix(self, angle):
        """
        Generate a rotation matrix for a given angle around the Y-axis.
        """
        rotation_3x3 = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        mat = np.eye(4)
        mat[:3, :3] = rotation_3x3
        return mat
    
    def _find_bounding_box(self, rgba_array):
        """
        Find the bounding box of the non-transparent part of the RGBA image.
        """
        rows = np.any(rgba_array[:, :, 3], axis=1)
        cols = np.any(rgba_array[:, :, 3], axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

    def _recover_original_image(
        self, 
        rgba_image: np.array, 
        rmin: int, 
        rmax: int, 
        cmin: int, 
        cmax: int
    ):
        """
        Move and scale image back to its original location
            PIL.Image: The recovered original image.
        """
        transformation = self.transformation_data
        
        scale_factor = 1 / transformation["scale_factor"]
        scaled_size = (
            int((cmax - cmin + 1) * scale_factor), 
            int((rmax - rmin + 1) * scale_factor)
        )
        
        object_region = rgba_image[rmin:rmax+1, cmin:cmax+1]
        scaled_image_region = Image.fromarray(object_region).resize(scaled_size)
    
        output_shape = (self.cfg.width, self.cfg.height, rgba_image.shape[-1])
        output_array = np.zeros(output_shape)

        org_dims = transformation["image_dimensions"]
    
        offset_row = int(transformation["translation"][1] / org_dims["height"] * self.cfg.height + 
                        (rmin - self.original_location[0]))
        offset_col = int(transformation["translation"][0] / org_dims["width"] * self.cfg.width + 
                        (cmin - self.original_location[1]))
        
        start_row = self.start_position[0] + offset_row
        start_col = self.start_position[1] + offset_col
       
        obj = np.array(scaled_image_region)[max(0, -start_row):self.cfg.height - start_row,
                 max(0, -start_col):self.cfg.width - start_col]
       
        start_row = max(start_row, 0)
        start_col = max(start_col, 0)
        output_array[start_row:start_row + obj.shape[0], 
                    start_col:start_col + obj.shape[1]] = obj
       
        return Image.fromarray(output_array.astype(np.uint8))

    
    def _add_mesh_to_scene(self, scene, mesh):
        """
        Add a mesh or meshes from a trimesh.Scene object to a pyrender scene.
        """
        if isinstance(mesh, trimesh.Scene):
            meshes = mesh.geometry.values()
        else:
            meshes = [mesh]

        for mesh in meshes:
            if isinstance(mesh, trimesh.Trimesh):
                cur_mesh = pyrender.Mesh.from_trimesh(mesh)
                cur_mesh.primitives[0].material.doubleSided = True
                scene.add(cur_mesh)
            else:
                raise TypeError(f"Expected a Trimesh or a Scene, got a {type(mesh)}")