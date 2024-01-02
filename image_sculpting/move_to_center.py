import argparse
import json
import os
from PIL import Image
import numpy as np
from dataclasses import dataclass, field
import matplotlib.pyplot as plt 
from loguru import logger as lg 
from skimage import io 

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--input_image', type=str, default="./control3d_data/dog3/mask0.png")
parser.add_argument('--output_image', type=str)
parser.add_argument('--log_file', type=str, default="./sculpting_data/data.json")
parser.add_argument('--percent_area', type=float, default=2e-1)
args = parser.parse_args()

filename, file_extension = os.path.splitext(args.input_image)
args.output_image = f"{filename}{file_extension}"

class ReCenter:
    def __init__(self):
        pass 

    def _scale_to_target_area(
        self,
        rgba_array, 
        target_area_ratio
    ):
        alpha_channel = rgba_array[:, :, 3]
        current_area = np.sum(alpha_channel > 0)
        total_area = rgba_array.shape[0] * rgba_array.shape[1]
        target_area = total_area * target_area_ratio

        scale_factor = 1.
        rmin, rmax, cmin, cmax = self._find_bounding_box(rgba_array)
        if current_area >= target_area: 
            return scale_factor, rgba_array, rmin, cmin

        
        object_region = rgba_array[rmin:rmax+1, cmin:cmax+1]
        
        scale_factor = np.sqrt(target_area / current_area)
        max_scale_factor_height = rgba_array.shape[0] / (rmax - rmin + 1)
        max_scale_factor_width = rgba_array.shape[1] / (cmax - cmin + 1)
        
        scale_factor = min(scale_factor, max_scale_factor_height, max_scale_factor_width)
    
        scaled_image_region = Image.fromarray(object_region).resize(
            (int((cmax - cmin + 1) * scale_factor), int((rmax - rmin + 1) * scale_factor))
        )
        output_array = np.zeros_like(rgba_array)
        start_row = (rgba_array.shape[0] - scaled_image_region.height) // 2
        start_col = (rgba_array.shape[1] - scaled_image_region.width) // 2
        
        output_array[start_row:start_row + scaled_image_region.height, start_col:start_col + scaled_image_region.width] = scaled_image_region
        return scale_factor, output_array, rmin, cmin
    
    @staticmethod
    def numpy_to_native(value):
        if isinstance(value, (np.int64, np.int32)):
            return int(value)
        elif isinstance(value, (np.float64, np.float32)):
            return float(value)
        elif isinstance(value, list):
            return [ReCenter.numpy_to_native(v) for v in value]
        elif isinstance(value, dict):
            return {k: ReCenter.numpy_to_native(v) for k, v in value.items()}
        return value

    def move_object_to_center(
        self,
        input_path, 
        output_path
    ):
        rgba_image = Image.open(input_path)
        rgba_array = np.array(rgba_image)
    
        rmin, rmax, cmin, cmax = self._find_bounding_box(rgba_array)
       
        center_x = (cmin + cmax) // 2
        center_y = (rmin + rmax) // 2
        image_width, image_height = rgba_image.size
        x_offset, y_offset = (image_width // 2) - center_x, (image_height // 2) - center_y
        new_image = Image.new('RGBA', (image_width, image_height), (0, 0, 0, 0))
        
        new_image.paste(rgba_image, (x_offset, y_offset), rgba_image)
        new_image = np.array(new_image)
    
        (   
            scale_factor, 
            new_image,
            rmin, 
            cmin
        ) = self._scale_to_target_area(new_image, args.percent_area)
        io.imsave(output_path, new_image)
        rgba_image.close()
        data = {}
        if os.path.exists(args.log_file):
            with open(args.log_file, 'r') as json_file:
                try:
                    data = json.load(json_file)
                except json.JSONDecodeError:
                    pass
        input_filename = os.path.basename(input_path)
        transformation_data = {
            "translation": [ReCenter.numpy_to_native(-x_offset), ReCenter.numpy_to_native(-y_offset)],
            "scale_factor": ReCenter.numpy_to_native(scale_factor),
            "image_dimensions": {
                "width": ReCenter.numpy_to_native(rgba_array.shape[1]),
                "height": ReCenter.numpy_to_native(rgba_array.shape[0])
            },
            "position": [ReCenter.numpy_to_native(rmin), 
                         ReCenter.numpy_to_native(cmin)]
        }
        
        data["/".join(args.output_image.split('/')[-2:])] = transformation_data
        with open(args.log_file, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def _find_bounding_box(self, rgba_array):
        rows = np.any(rgba_array[:, :, 3], axis=1)
        cols = np.any(rgba_array[:, :, 3], axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax

@lg.catch
def main():
    recenter = ReCenter()
    recenter.move_object_to_center(args.input_image, args.output_image)
   
if __name__ == "__main__":
    main()