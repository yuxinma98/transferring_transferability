#!/bin/bash

declare -A model_args
model_args=( ["reduced"]=4 ["unreduced"]=4 ["simple"]=15 ["ign"]=5 ["ign_anydim"]=9)

for model in "${!model_args[@]}"
do
    hidden_channels=${model_args[$model]}
    echo "Running size_generalizability.py with model argument: $model and hidden channels: $hidden_channels"
    CUDA_VISIBLE_DEVICES=3 python size_generalizability.py --model "$model" --graph_model "SBM_Gaussian" --hidden_channels "$hidden_channels" --task "triangle"
done