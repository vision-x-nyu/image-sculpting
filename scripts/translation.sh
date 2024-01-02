#!/bin/bash

declare -A DATA_MASK_POSE_MAP
declare -A DATA_PROMPT_MAP


DATA_MASK_POSE_MAP["dog1"]="mask0 pose0"
DATA_MASK_POSE_MAP["worm1"]="mask0 pose0"
DATA_MASK_POSE_MAP["car1"]="mask0 pose0"

DATA_PROMPT_MAP["dog1"]="dog running"
DATA_PROMPT_MAP["worm1"]="worm"
DATA_PROMPT_MAP["car1"]="truck"

for DATA in "${!DATA_MASK_POSE_MAP[@]}"; do
    MASK_POSE=${DATA_MASK_POSE_MAP[$DATA]}
    MASK_POSES=($MASK_POSE)  # Convert string to array
    MASK=""
    for val in "${MASK_POSES[@]}"; do
        if [[ $val == mask* ]]; then
            MASK=$val
        else
            POSE=$val
            PROMPT=${DATA_PROMPT_MAP[$DATA]}
            INSTANCE_PROMPT="a photo of sks $PROMPT"
            GPU=1
            N_VIEWS=1
           
            echo "Running with DATA=$DATA, POSE=$POSE, PROMPT=$PROMPT, INSTANCE_PROMPT=$INSTANCE_PROMPT"

            # Run the main command
            CUDA_VISIBLE_DEVICES=$GPU python enhancement.py \
                --config configs/image_sculpting.yaml \
                --test \
                --gpu $GPU \
                name=results/translation/"$DATA" \
                system.renderer.deformed_mesh="./runs/sculpting_data/$DATA/"$MASK"_resize.png/mesh/$POSE.obj" \
                system.renderer.original_mesh="./runs/sculpting_data/$DATA/"$MASK"_resize.png/mesh/model.obj" \
                system.model.instance_prompt="$INSTANCE_PROMPT" \
                system.model.lora_weights="./runs/dreambooth_ckpts/$DATA" \
                data.n_views=1 \
                system.inpainting.bg_path="./sculpting_data/$DATA/bg.png" \
                system.inversion.prompt="a photo of $PROMPT"
        fi
    done
done
