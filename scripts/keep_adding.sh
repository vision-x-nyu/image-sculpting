#!/bin/bash

declare -A DATA_MASK_POSE_MAP
declare -A DATA_PROMPT_MAP

DATA_MASK_POSE_MAP["cake2"]="mask0 model"
DATA_PROMPT_MAP["cake2"]="cake"

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
            for i in {1..4}; do  # Loop from 1 to 4
                INSTANCE_PROMPT="a photo of sks $PROMPT"
                GPU=1
                N_VIEWS=1
                OBJ="runs/3d_models/$PROMPT/${i}/$PROMPT.obj"
                echo "Running with DATA=$DATA, POSE=$POSE, PROMPT=$PROMPT, INSTANCE_PROMPT=$INSTANCE_PROMPT, OBJ=$OBJ"
                
                # Run the main command
                CUDA_VISIBLE_DEVICES=$GPU python enhancement.py \
                    --config configs/image_sculpting.yaml \
                    --test \
                    --gpu $GPU \
                    name=results/adding/"$DATA"_$i \
                    system.renderer.deformed_mesh="./runs/sculpting_data/$DATA/"$MASK"_resize.png/mesh/$POSE.obj" \
                    system.renderer.original_mesh="./runs/sculpting_data/$DATA/"$MASK"_resize.png/mesh/model.obj" \
                    system.model.instance_prompt="$INSTANCE_PROMPT" \
                    system.model.lora_weights="./runs/dreambooth_ckpts/$DATA" \
                    data.n_views=$N_VIEWS \
                    system.inpainting.bg_path="./sculpting_data/$DATA/bg.png" \
                    system.inversion.prompt="a photo of $PROMPT"\
                    system.renderer.inserted_object=$OBJ
            done
        fi
    done
done
