import argparse
import numpy as np
from typing import Union
from train import train


def nrange(value: Union[str, list]) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return list(np.arange(*map(float, value.split(":"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="Cora", choices=["Cora", "PubMed"], help="Dataset to use"
    )
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials to run")
    # GNN parameters
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--hidden_channels", type=int, default=50, help="Number of hidden channels")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum number of epochs")

    args = parser.parse_args()
    params = {
        # logger parameters
        "project": "anydim_transferability",  # wandb project name
        "name": "GNN_performance",  # wandb run name
        "logger": True,  # whether to use wandb to log
        "log_checkpoint": "best",  # whether to log model checkpoint in wandb
        "log_model": None,  # whether to log gradient/parameters in wandb
        "log_dir": "log/performance",  # directory to save logs
        # data parameters
        "dataset": args.dataset,
        "batch_size": 20,  # batch size to use in trasnferability experiment (set smaller to avoid memory issues)
        # model parameters
        "model": {
            "model_name": "GNN",  # choices ["GNN", "IGN"]
            "hidden_channels": args.hidden_channels,
            "num_layers": args.num_layers,
            "task": "classification",
            "simple": False,  # choice: [True, False, "Laplacian"]
            "reduced": False,
        },
        # training parameters
        "lr": args.lr,
        "lr_patience": 10,
        "weight_decay": 0.1,
        "max_epochs": args.max_epochs,
        "training_seed": 42,
    }
    for i in range(args.num_trials):
        params["training_seed"] = i
        model = train(params)
