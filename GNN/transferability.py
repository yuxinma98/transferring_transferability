import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from typing import Union
from data import SubsampledDataset
from train import train, GNNTrainingModule


def nrange(value: Union[str, list]) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return list(np.arange(*map(float, value.split(":"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment set-ups
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        choices=["Cora", "PubMed", "SBM_Gaussian"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=200,
        help="In the transferability experiment, for each graph size n, how many samples to generate from the step graphon",
    )
    parser.add_argument(
        "--reference_graph_size", type=int, default=int(1e4), help="Reference graph size"
    )
    parser.add_argument(
        "--log_n_range", type=nrange, default="1:2.6:0.2", help="Range of log n to consider"
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
        "log_model": None,  # whether to log gradient/parameters in wandb
        "log_dir": "log/transferability",  # directory to save logs
        # data parameters
        "dataset": args.dataset,
        "batch_size": 2,  # batch size to use in trasnferability experiment (set smaller to avoid memory issues)
        # model parameters
        "model": {
            "hidden_channels": args.hidden_channels,
            "num_layers": args.num_layers,
            # "task": "classification",
            "model": "unreduced",  # choice in ["simple", "reduced", "unreduced", "ign"]
        },
        # training parameters
        "lr": args.lr,
        "lr_patience": 10,
        "weight_decay": 0.1,
        "max_epochs": args.max_epochs,
        "training_seed": 42,
    }

    # train model
    model = train(params)
    # model = GNNTrainingModule(params)
    model.eval()

    # transferability experiment
    # subsampled_data = SubsampledDataset(
    #     "data/", model, args.dataset, args.n_samples, args.reference_graph_size
    # )
    # test_loader = DataLoader(subsampled_data, batch_size=params["batch_size"], shuffle=False)
    # with torch.no_grad():
    #     reference_out = []
    #     for batch in test_loader:
    #         out = model(batch).detach()  # batch_size x n x D_out
    #         projected_out = out[
    #             torch.arange(out.shape[0]).unsqueeze(-1), batch.indices, :
    #         ]  # batch_size x N x D_out
    #         reference_out.append(projected_out)
    #     reference_out = torch.cat(reference_out, dim=0)  # n_samples x N x D_out
    # reference_out = reference_out.mean(dim=0)  # N x D_out

    n_range = np.power(10, args.log_n_range).astype(int)
    errors_mean = np.zeros_like(n_range, dtype=float)
    errors_std = np.zeros_like(n_range, dtype=float)
    for i, n in enumerate(n_range):
        # sample smaller graphs and graph signals from the step graphon
        subsampled_data = SubsampledDataset("data/", model, args.dataset, args.n_samples, n)
        test_loader = DataLoader(subsampled_data, batch_size=params["batch_size"], shuffle=False)
        # compute GNN output on subsampled graphs
        errors = []
        target = subsampled_data.target  # N x D_out
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch).detach()  # batch_size x n x D_out
                projected_out = out[
                    torch.arange(out.shape[0]).unsqueeze(-1), batch.indices, :
                ]  # batch_size x N x D_out
                errors.append(
                    torch.norm(projected_out - target, p=2, dim=1) / np.sqrt(subsampled_data.n)
                )
                print(target.max(), target.min(), projected_out.max(), projected_out.min())
        errors = torch.cat(errors, dim=0)  # n_samples x D_out
        errors = errors.max(dim=1)[0]  # n_samples
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
    plt.ylabel("$\max_j \|f_n(A_n,X_n)_{:j} - f_m(A,X)_{:j}\|_2$")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    image_path = os.path.join(
        params["log_dir"], f"transferability_{args.dataset}_{params['model']['model']}.png"
    )
    plt.savefig(image_path)
