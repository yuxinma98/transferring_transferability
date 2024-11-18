import argparse
import wandb
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from typing import Union
from data import SubsampledDataset
from train import train


def nrange(value: Union[str, list]) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return list(np.arange(*map(float, value.split(":"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment set-ups
    parser.add_argument(
        "--dataset", type=str, default="Cora", choices=["Cora", "PubMed"], help="Dataset to use"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1000,
        help="In the transferability experiment, for each graph size n, how many samples to generate from the step graphon",
    )
    parser.add_argument(
        "--reference_graph_size", type=int, default=int(1e4), help="Reference graph size"
    )
    parser.add_argument(
        "--log_n_range", type=nrange, default="1:3:0.2", help="Range of log n to consider"
    )
    # GNN parameters
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--hidden_channels", type=int, default=50, help="Number of hidden channels")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum number of epochs")

    args = parser.parse_args()
    params = {
        # logger parameters
        "project": "anydim_transferability",  # wandb project name
        "name": "GNN_transferability",  # wandb run name
        "logger": True,  # whether to use wandb to log
        "log_checkpoint": False,  # whether to log model checkpoint in wandb
        "log_model": None,  # model to log in wandb
        "log_dir": "log/transferability",  # directory to save logs
        # data parameters
        "dataset": args.dataset,
        "batch_size": 20,  # batch size to use in trasnferability experiment (set smaller to avoid memory issues)
        # model parameters
        "model": {
            "hidden_channels": args.hidden_channels,
            "num_layers": args.num_layers,
            "out_channels": 7,  # classification into 7 classes
            "task": "classification",
            "reduced": False,
        },
        # training parameters
        "lr": args.lr,
        "lr_patience": 10,
        "weight_decay": 0.1,
        "max_epochs": args.max_epochs,
        "training_seed": 42,
    }
    trainer, model = train(params)
    model.eval()

    # transferability experiment
    with torch.no_grad():
        # use GNN output on full graph as reference
        predict_loader = DataLoader([model.data], batch_size=1, shuffle=False)
        subsampled_data = SubsampledDataset("data/", args.dataset, 1, args.reference_graph_size)
        test_loader = DataLoader(
            [data for data in subsampled_data], batch_size=params["batch_size"], shuffle=False
        )
        out = trainer.predict(model, test_loader, ckpt_path="best")[0]
        reference_out = out.mean(dim=0)

        n_range = np.power(10, args.log_n_range).astype(int)
        errors_mean = np.zeros_like(n_range, dtype=float)
        errors_std = np.zeros_like(n_range, dtype=float)
        for i, n in enumerate(n_range):
            # sample smaller graphs and graph signals from the step graphon
            subsampled_data = SubsampledDataset("data/", args.dataset, args.n_samples, n)
            test_loader = DataLoader(
                [data for data in subsampled_data], batch_size=params["batch_size"], shuffle=False
            )
            # compute GNN output on subsampled graphs
            out = trainer.predict(model, test_loader, ckpt_path="best")
            out = torch.cat(out, dim=0)

            # compute errors from the n_samples small graphs; record mean and std of errors
            errors = torch.abs(out - reference_out)
            errors_mean[i] = errors.mean().item()
            errors_std[i] = errors.std().item()

    # plot and log transferability results
    plt.figure()
    plt.errorbar(
        n_range, errors_mean, errors_std, fmt="o", capsize=3, markersize=5, label="Reduced model"
    )
    y = n_range ** (-0.5) * n_range[0] ** (0.5) * errors_mean[0]
    plt.plot(n_range, y, label="$n^{-1/2}$")
    plt.xlabel("Set size $n$")
    plt.ylabel("$|f_n(x) - f_m(x)|$")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    image_path = os.path.join(params["log_dir"], f"transferability_{args.dataset}.png")
    plt.savefig(image_path)
    if params["logger"]:
        wandb.log({"transferability": wandb.Image(str(image_path))})
        wandb.finish()
