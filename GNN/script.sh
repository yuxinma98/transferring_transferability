sample_fractions=(0.1 0.2 0.3 0.4 0.5)
for fraction in "${sample_fractions[@]}"; do
    echo "Running main.py with sample_fraction=$fraction"
    CUDA_VISIBLE_DEVICES=1 python main.py --sample_fraction "$fraction" --num_layers 2 --lr 5e-4 --hidden_channels 16 --max_epochs 600
done