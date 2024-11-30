#!/bin/bash

model_args=("reduced" "unreduced" "simple" "ign")

for arg in "${model_args[@]}"
do
    echo "Running size_generalizability.py with model argument: $arg"
    python size_generalizability.py --model "$arg"
done