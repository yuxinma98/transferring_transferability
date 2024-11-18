import torch
import argparse
import os
import numpy as np
import h5py
import wandb
from typing import Union
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from train import train
from data import SubsampledDataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def nrange(value: Union[str, list]) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return list(np.arange(*map(float, value.split(":"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment set-up
    parser.add_argument("--training_size", type=int, default=500, help="Set size of training data")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument(
        "--reference_set_size", type=int, default=int(1e8), help="Reference set size"
    )
    parser.add_argument(
        "--log_n_range",
        type=nrange,
        default="1:4:0.2",
        help="Log range of set sizes",
    )

    # DeepSet model parameters
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=50)
    parser.add_argument("--set_channels", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=1000)

    args = parser.parse_args()

    params = {
        # logger parameters
        "project": "anydim_transferability",
        "name": "deepset_transferability",
        "logger": True,
        "log_checkpoint": False,
        "log_model": None,
        "log_dir": os.path.join(CURRENT_DIR, "log/transferability"),
        # data parameters
        "data_dir": os.path.join(CURRENT_DIR, "generator/data"),
        "training_size": args.training_size,
        "batch_size": 128,
        # model parameters
        "task_id": 1,
        "model": {
            "hidden_channels": args.hidden_channels,
            "set_channels": args.set_channels,
            "feature_extractor_num_layers": args.num_layers,
            "regressor_num_layers": args.num_layers,
            "num_layers": args.num_layers,
            "normalized": True,
        },
        # training parameters
        "lr": args.lr,
        "lr_patience": 50,
        "weight_decay": 0.1,
        "max_epochs": args.max_epochs,
        "training_seed": 0,
    }
    if not os.path.exists(params["log_dir"]):
        os.makedirs(params["log_dir"])
    if not os.path.exists(params["data_dir"]):
        os.makedirs(params["data_dir"])

    model = train(params)
    model.eval()

    # Load distribution parameter of data
    with h5py.File(os.path.join(params["data_dir"], "task1/matrix_A.mat"), "r") as f:
        A = torch.tensor(f["A"][()], dtype=torch.float32)
        cov = A @ A.T

    # Take a large sample
    multivariate_normal = torch.distributions.MultivariateNormal(torch.zeros(2), cov)
    X_reference = multivariate_normal.sample((args.reference_set_size,)).unsqueeze(0)
    with torch.no_grad():
        y_reference = float(model(X_reference).mean(dim=0))

    # subsampling from the large sample, and compute the error
    n_range = np.power(10, args.log_n_range).astype(int)
    errors_mean = np.zeros_like(n_range, dtype=float)
    errors_std = np.zeros_like(n_range, dtype=float)
    for i, n in enumerate(n_range):
        subsampled_data = SubsampledDataset(X_reference, n_samples=args.n_samples, set_size=n)
        loader = DataLoader(subsampled_data, batch_size=params["batch_size"], shuffle=False)
        with torch.no_grad():
            y = []
            for batch in loader:
                y.append(model(batch))
        y = torch.cat(y, dim=0)
        error = torch.abs(y - y_reference)
        errors_mean[i] = float(error.mean(dim=0).squeeze())
        errors_std[i] = float(error.std(dim=0).squeeze())
    wandb.finish()

    # plot
    plt.figure()
    plt.errorbar(n_range, errors_mean, errors_std, fmt="o", capsize=3, markersize=5)
    reference = n_range ** (-0.5) * n_range[0] ** (0.5) * errors_mean[0]
    plt.plot(n_range, reference, label="$n^{-0.5}$")
    plt.xlabel("Set size $n$")
    plt.ylabel("$|f_n(x) - f_m(x)|$")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(os.path.join(params["log_dir"], "transferability.png"))
