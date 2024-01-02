# Image Sculpting: Precise Object Editing with 3D Geometry Control

### [Project Page](https://image-sculpting.github.io/) | [Video](https://youtu.be/qdk6sVr47MQ?si=5CFkkqVLhMb76guF) | [Paper]() | [Data](https://drive.google.com/drive/folders/1VeopyGxegiQg-Skqisluvko9ed9y3rXl?usp=drive_link)

Authors: [Jiraphon Yenphraphai](https://domejiraphon.github.io/)<sup>1</sup>, [Xichen Pan](https://xichenpan.com/)<sup>1</sup>, [Sainan Liu](https://www.linkedin.com/in/sainan-stephanie-liu)<sup>2</sup>, [Daniele Panozzo](https://cims.nyu.edu/gcl/daniele.html)<sup>1</sup>, [Saining Xie](https://www.sainingxie.com/)<sup>1</sup>  
Affiliations: <sup>1</sup>New York University, <sup>2</sup>Intel Labs  


[![Image Sculpting](https://i.imgur.com/LZ3Fc0e.jpeg)](https://youtu.be/ZLAE8pTG5g8?si=2P-zPHatOf9MBGS1)

We present Image Sculpting, a new framework for editing 2D images by incorporating tools from 3D geometry and graphics. This approach differs markedly from existing methods, which are confined to 2D spaces and typically rely on textual instructions, leading to ambiguity and limited control. Image Sculpting converts 2D objects into 3D, enabling direct interaction with their 3D geometry. Post-editing, these objects are re-rendered into 2D, merging into the original image to produce high-fidelity results through a coarse-to-fine enhancement process. The framework supports precise, quantifiable, and physically-plausible editing options such as pose editing, rotation, translation, 3D composition, carving, and serial addition. It marks an initial step towards combining the creative freedom of generative models with the precision of graphics pipelines.
## Table of contents
-----
- [Setup](#setup)
- [Running Our Images](#running-our-images)
    - [Pose Editing](#pose-editing)
    - [Rotation](#rotation)
    - [Translation](#translation)
    - [3D Composition](#3d-composition)
    - [Carving](#carving)
    - [Serial Addition](#serial-addition)
- [Using Your Own Data](#using-your-own-data)
- [Citation](#citation)
------

## Setup
We have included a `requirements.txt` file to help you in set up a virtual environment

```shell
python -m venv venv_img_sculpting
source venv_img_sculpting/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```
This code is borrowed from [threestudio](https://github.com/threestudio-project/threestudio) and has been tested on an NVIDIA RTX 4090 using CUDA 12.0. If you encounter any installation issues, please refer to their repository for guidance.
## Running our images
We have provided the reconstructed meshes along with its 3D editing in [google drive](https://drive.google.com/drive/folders/1cjUyssJaarYNpmY0pKxDGA4aOAlqZAad?usp=sharing), supporting the following of our capabilities. Download the `zip file` from Google Drive and place it in the `image_sculpting`.
```bash
git clone https://github.com/domejiraphon/threestudio
cd image-sculpting
unzip runs.zip
```
### 1. Pose Editing
To edit the pose, run the following command:
```bash
bash scripts/pose_editing.sh
```
### 2. Rotation
To edit the orientation of an object, run the following command:
```bash
bash scripts/pose_rotation.sh
```

### 3. Translation
To edit the orientation of an object, run the following command:
```bash
bash scripts/translation.sh
```

### 4. 3D Composition
Example scripts for training, evaluating, and rendering can be found in
`scripts/composition`.
To add new 3D model to the scene, you can run the example script
```bash
bash scripts/composition/joker_tiger.sh
```

### 5. Carving
To carve the mesh, run the following command:
```bash
bash scripts/carving.sh
```

### 6. Serial Addition
To keep adding more 3D models, run the following command:
```bash
bash scripts/keep_adding.sh
```

## Using your own image
### 1. De-Rendering
We use the implementation provided by [threestudio](https://github.com/threestudio-project/threestudio?tab=readme-ov-file#zero-1-to-3-) to generate a 3D model from a single image. This step involves [Zero-1-to-3](https://zero123.cs.columbia.edu/) and [SDS](https://dreamfusion3d.github.io/). For installation instructions, you can follow the guidelines provided in their documentation. We have observed that recentering and scaling the image are necessary steps for the Zero-1-to-3 to obtain a good reconstruction. The following code will do that for you.
#### 1.1 Preprocessing
##### 1.1.1 Remove the background of the image which can be done using [Clipdrop](https://clipdrop.co/remove-background)
#####  1.1.2 Create a folder and save it to `sculpting_data/[your_folder]`
```bash
python image_sculpting/move_to_center.py \
                --input_image="$FILE" 
```
#### 1.2 3D reconstruction.
**Zero 1-to-3 Installation**

Download pretrained Zero 1-to-3 XL weights into `load/zero123`:

```sh
cd load/zero123
wget https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt
```
Run Zero-1-to-3
```bash
CUDA_VISIBLE_DEVICES=0 python launch.py \
    --config configs/zero123.yaml\
    --train \
    --gpu 0 \
    data.image_path="$FILE" \
    exp_root_dir="./runs"
```
#### 1.3 Extract the mesh
```bash
CUDA_VISIBLE_DEVICES=0 python launch.py \
    --config runs/sculpting_data/$FILE/configs/parsed.yaml \
    --export \
    --gpu 0 \
    resume=runs/sculpting_data/$FILE/ckpts/last.ckpt \
    system.geometry.isosurface_method=mc-cpu \
    system.geometry.isosurface_resolution=128
```
You can also try [Stable Zero-1-to-3](https://github.com/threestudio-project/threestudio/tree/main?tab=readme-ov-file#stable-zero123) for better reconstruction.
### 2. Deformation
Now that you have the reconstructed mesh, you can try the following 3D editing steps.
#### 2.1 Pose Editing
If you're interested in pose editing, we recommend constructing the bone structure and deforming the 3D mesh accordingly. For guidance, follow the instructions in this [tutorial](https://youtu.be/jj4IZ5iEzAo?si=HS_fk4sAS3LP5PMz). While other deformation methods can also be used, employing bones offers the most intuitive and physics-aware approach.

#### 2.2 Rotation
No additional steps are required for rotation.

#### 2.3 Translation
To perform translation, move the mesh within an application like Blender.

#### 2.4 Carving
For carving, we suggest using the boolean operator in Blender. Learn how to do this from this [tutorial](https://youtu.be/62rDhWB6O-0?si=1WGXSZ7m6bGzHZBk).

#### 2.5 3D Composition and Serial Addition
For 3D composition, simply add the 3D model to the scene. This approach is also applicable for serial addition.


### 3. Generative Enhancement
Our method requires fine-tuning to capture the texture of the input image. We follow the DreamBooth implementation provided by [diffusers](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sdxl.md). After your model has completed training, you can refer to the [Running Our Images](#running-our-images) section for examples on how to execute each of our capabilities or the example below.

```bash
export N_VIEWS=4 # Number of views for a complete circle rotation.
export IMG_PATH="test/img.png" # input image path
export LORA_PATH="./runs/dreambooth_ckpts/test" # path to lora
export BG_PATH="./sculpting_data/test/bg.png" # path to the inpainting background
export INSTANCE_PROMPT="a photo of sks test" # Prompt used during the fine-tuning of the instance image.
export INVERSION_PROMPT="a photo of test" # Prompt used during the fine-tuning of the class image.
export DEFORMED_MESH_PATH="./runs/sculpting_data/$IMG_PATH/mesh/model.obj" # Deformed mesh path
export ORG_MESH_PATH="./runs/sculpting_data/$IMG_PATH/mesh/model.obj" # Original mesh path

CUDA_VISIBLE_DEVICES=0 python enhancement.py \
                --config configs/image_sculpting.yaml \
                --test \
                --gpu 0 \
                name=results/output \
                system.renderer.deformed_mesh=$DEFORMED_MESH_PATH\
                system.renderer.original_mesh=$ORG_MESH_PATH \
                system.model.instance_prompt="$INSTANCE_PROMPT" \
                system.model.lora_weights=$LORA_PATH \
                data.n_views=$N_VIEWS \
                system.inpainting.bg_path=$BG_PATH \
                system.inversion.prompt="$INVERSION_PROMPT"
```

## License

This project is under the MIT license. See [LICENSE](LICENSE) for details.

## Citation

```
@inproceedings{Yenphraphai2023imagesculpting,
    author = {Yenphraphai, Jiraphon and Pan, Xichen and Liu, Sainan and Panozzo, Daniele and Xie, Saining},
    title = {Image Sculpting: Precise Object Editing with 3D Geometry Control},
    journal = {arXiv preprint}, 
    year = {2023},
}
```