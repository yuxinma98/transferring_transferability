import argparse
import os
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import torch
from scipy.optimize import linear_sum_assignment
from torch_geometric.loader import DataLoader
from tqdm import tqdm
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
        "--model",
        type=str,
        default="reduced",
        choices=["simple", "reduced", "unreduced", "ign", "ign_anydim"],
    )
    parser.add_argument(
        "--graph_model", type=str, default="Sociality", choices=["ER", "SBM", "Sociality", "spiked"]
    )
    parser.add_argument(
        "--task",
        type=str,
        default="conditional_triangle",
        choices=["degree", "triangle", "4-cycle", "conditional_triangle"],
    )
    parser.add_argument("--training_graph_size", type=int, default=50)
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="In the transferability experiment, for each graph size n, how many samples to generate from the step graphon",
    )
    # GNN parameters
    parser.add_argument("--num_layers", type=int, default=5, help="Number of GNN layers")
    parser.add_argument("--hidden_channels", type=int, default=4, help="Number of hidden channels")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum number of epochs")

    args = parser.parse_args()
    params = {
        # logger parameters
        "project": "anydim_transferability",  # wandb project name
        "name": "GNN_transferability",  # wandb run name
        "logger": True,  # whether to use wandb to log
        "log_checkpoint": True,  # whether to log model checkpoint in wandb
        "log_model": None,  # whether to log gradient/parameters in wandb
        "log_dir": "log/transferability",  # directory to save logs
        # data parameters
        "n_graphs": 5000,  # size of training dataset
        "n_nodes": args.training_graph_size,
        "graph_model": args.graph_model,
        "task": args.task,
        "data_dir": "/export/canton/data/yma93/anydim_transferability/GNN_size_generalizability",
        "val_fraction": 0.2,
        "test_fraction": 0.2,
        "batch_size": 128,  # batch size to use in trasnferability experiment (set smaller to avoid memory issues)
        "data_seed": 1,
        # model parameters
        "model": {
            "hidden_channels": args.hidden_channels,
            "num_layers": args.num_layers,
            "model": args.model,  # choice in ["simple", "reduced", "unreduced", "ign", "ign_anydim"]
            "bias": True,
        },
        # training parameters
        "lr": args.lr,
        "lr_patience": 10,
        "weight_decay": 0.1,
        "max_epochs": args.max_epochs,
        "training_seed": 1,
    }
    if not os.path.exists(params["log_dir"]):
        os.makedirs(params["log_dir"])
    fname = f"results_{args.graph_model}_{args.task}_{args.model}.json"

    try:
        with open(os.path.join(params["log_dir"], fname), "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    n_range = np.array([50, 100, 250, 500, 1000, 2500, 5000, 10000])
    if not (
        all(str(n) in results for n in n_range)
        and all(len(results[str(n)]) == args.n_samples for n in n_range)
    ):
        # train model
        model = train(params)
        model.eval()
        # model = GNNTrainingModule(params)
        # model.prepare_data()

        for i, n in enumerate(n_range):
            print(f"running for n = {n}")
            results.setdefault(str(n), [])
            n_samples = args.n_samples - len(results[str(n)])
            if n_samples == 0:
                continue
            # sample smaller graphs and graph signals from the step graphon
            subsampled_data = SubsampledDataset(
                model=model,
                n_samples=n_samples,
                n_nodes=n,
            )
            test_loader = DataLoader(subsampled_data, batch_size=1, shuffle=False)

            # find target
            lcm = math.lcm(n, subsampled_data.N)
            target = subsampled_data.target.repeat(
                1, lcm // subsampled_data.N, 1
            )  # 1 * lcm * d_out
            target, _ = target.sort(dim=1)
            # compute GNN output on subsampled graphs
            errors = []
            with torch.no_grad():
                for batch in tqdm(test_loader):
                    out = model(batch).detach()  # batch_size x n
                    out_transformed = out.repeat(1, lcm // n, 1)  # batch_size x lcm x d_out
                    # if target.shape[-1] == 1:
                    out_transformed, _ = out_transformed.sort(dim=1)
                    errors.append(torch.norm(out_transformed - target, p=2, dim=-2) / np.sqrt(lcm))
                    # else:  # require solving the assignment problem:
                    for j in range(len(batch)):
                        row_idx, col_idx = linear_sum_assignment(
                            -torch.matmul(
                                out_transformed[j, :, :], target.squeeze(0).transpose(-2, -1)
                            )
                        )  # this maximize Tr(x @ y.t @ P) over permutation P
                        errors.append(
                            torch.norm(
                                out_transformed[j, row_idx, :] - target[0, col_idx, :],
                                p=2,
                                dim=-2,
                            ).unsqueeze(0)
                            / np.sqrt(lcm)
                        )
                errors = torch.concat(errors, dim=0)  # n_samples x D_out
                errors = errors.max(dim=1)[0]  # n_samples
                results[str(n)] += errors.tolist()
                with open(os.path.join(params["log_dir"], fname), "w") as f:
                    json.dump(results, f)

    errors_mean = np.zeros_like(n_range, dtype=float)
    errors_std = np.zeros_like(n_range, dtype=float)
    for i in range(len(n_range)):
        errors = np.array(results[str(n_range[i])])
        errors_mean[i] = errors.mean()
        errors_std[i] = errors.std()

    # plot and log transferability results
    plt.figure()
    plt.errorbar(n_range, errors_mean, errors_std, fmt="o", capsize=3, markersize=5)
    y = n_range ** (-0.5) * n_range[0] ** (0.5) * errors_mean[0]
    plt.plot(n_range, y, label="$n^{-1/2}$")
    plt.xlabel("Set size $n$")
    plt.ylabel("$\max_j \|f_n(A_n,X_n)_{:j} - f_N(A,X)_{:j}\|_2$")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    image_path = os.path.join(
        params["log_dir"],
        f"transferability_{args.graph_model}_{args.task}_{args.model}.png",
    )
    plt.savefig(image_path)
