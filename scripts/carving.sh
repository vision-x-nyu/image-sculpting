#!/bin/bash
declare -A DATA_MASK_POSE_MAP
declare -A DATA_PROMPT_MAP

DATA_MASK_POSE_MAP["pumpkin1"]="mask0 pose0"
DATA_MASK_POSE_MAP["cake2"]="mask0 pose0"
DATA_MASK_POSE_MAP["watermelon1"]="mask0 pose0"


DATA_PROMPT_MAP["pumpkin1"]="pumpkin"
DATA_PROMPT_MAP["watermelon1"]="watermelon"
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
            INSTANCE_PROMPT="a photo of sks $PROMPT"
            GPU=0
            N_VIEWS=1
            if [[ $DATA == "pumpkin1" || $DATA == "watermelon1" ]]; then
                INSTANCE_PROMPT="$INSTANCE_PROMPT jack o lantern"
            fi
          
            echo "Running with DATA=$DATA, POSE=$POSE, PROMPT=$PROMPT, INSTANCE_PROMPT=$INSTANCE_PROMPT"
            
            # Run the main command
            CUDA_VISIBLE_DEVICES=$GPU python enhancement.py \
                --config configs/image_sculpting.yaml \
                --test \
                --gpu $GPU \
                name=results/carving/"$DATA"_"$POSE" \
                system.renderer.deformed_mesh="./runs/sculpting_data/$DATA/"$MASK"_resize.png/mesh/$POSE.obj" \
                system.renderer.original_mesh="./runs/sculpting_data/$DATA/"$MASK"_resize.png/mesh/model.obj" \
                system.model.instance_prompt="$INSTANCE_PROMPT" \
                system.model.lora_weights="./runs/dreambooth_ckpts/$DATA" \
                data.n_views=$N_VIEWS \
                system.inpainting.bg_path="./sculpting_data/$DATA/bg.png" \
                system.inversion.prompt="a photo of $PROMPT"
        fi
    done
done
