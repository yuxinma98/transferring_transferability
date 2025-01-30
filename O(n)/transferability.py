import torch
import argparse
import os
import numpy as np
import h5py
from typing import Union
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from train import train
from data import SubsampledDataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ibm_colors = [
    "#648FFF",  # Blue
    "#785EF0",  # Purple
    "#DC267F",  # Pink
    "#FE6100",  # Orange
    "#FFB000",  # Yellow
]


def nrange(value: Union[str, list]) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return list(np.arange(*map(float, value.split(":"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment set-up
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument(
        "--reference_set_size", type=int, default=500, help="Set size of reference data"
    )
    parser.add_argument(
        "--n_range",
        type=nrange,
        default="500:5000:500",
        help="Range of set sizes",
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
        "name": "oids_transferability",
        "logger": True,
        "log_checkpoint": False,
        "log_model": None,
        "log_dir": os.path.join(CURRENT_DIR, "log/transferability"),
        # data parameters
        "data_dir": "/export/canton/data/yma93/anydim_transferability/OI-DS/",
        "point_cloud_size": 20,
        # model parameters
        "model": {
            "hidden_channels": args.hidden_channels,
            "set_channels": args.set_channels,
            "out_channels": args.out_channels,
            "feature_extractor_num_layers": args.num_layers,
            "regressor_num_layers": args.num_layers,
            "num_layers": args.num_layers,
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

    normalized_model, _, _ = train(params)
    params["model"]["normalized"] = False
    unnormalized_model, _, _ = train(params)

    normalized_model.eval()
    unnormalized_model.eval()

    # Take a large sample, use as generative model
    multivariate_normal = torch.distributions.MultivariateNormal(torch.zeros(2), cov)
    X_reference = multivariate_normal.sample((args.reference_set_size,)).unsqueeze(0)
    with torch.no_grad():
        normalized_y_reference = float(normalized_model(X_reference))
        unnormalized_y_reference = float(unnormalized_model(X_reference))

    # subsampling from step graphon to get X_n for a range of n
    n_range = np.array(args.n_range).astype(int)
    normalized_error_mean = np.zeros_like(n_range, dtype=float)
    normalized_error_std = np.zeros_like(n_range, dtype=float)
    unnormalized_error_mean = np.zeros_like(n_range, dtype=float)
    unnormalized_error_std = np.zeros_like(n_range, dtype=float)
    errors_std = np.zeros_like(n_range, dtype=float)
    for i, n in enumerate(n_range):
        subsampled_data = SubsampledDataset(X_reference, n_samples=args.n_samples, set_size=n)
        loader = DataLoader(subsampled_data, batch_size=params["batch_size"], shuffle=False)
        with torch.no_grad():
            normalized_y = []
            unnormalized_y = []
            for batch in loader:
                normalized_y.append(normalized_model(batch))
                unnormalized_y.append(unnormalized_model(batch))
        normalized_y = torch.cat(normalized_y, dim=0)
        unnormalized_y = torch.cat(unnormalized_y, dim=0)
        normalized_error = torch.abs(normalized_y - normalized_y_reference)
        unnormalized_error = torch.abs(unnormalized_y - unnormalized_y_reference)
        normalized_error_mean[i] = float(normalized_error.mean(dim=0).squeeze())
        normalized_error_std[i] = float(normalized_error.std(dim=0).squeeze())
        unnormalized_error_mean[i] = float(unnormalized_error.mean(dim=0).squeeze())
        unnormalized_error_std[i] = float(unnormalized_error.std(dim=0).squeeze())

    plt.rcParams.update({"font.size": 18})  # Adjust font size as needed
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed

    plt.errorbar(
        n_range,
        normalized_error_mean,
        errors_std,
        capsize=10,
        markersize=10,
        elinewidth=2,
        fmt="o",
        color=ibm_colors[0],
        label="Normalized DeepSet",
        linestyle="none",
    )
    plt.errorbar(
        n_range,
        unnormalized_error_mean,
        errors_std,
        capsize=10,
        markersize=10,
        elinewidth=2,
        fmt="o",
        color=ibm_colors[3],
        label="DeepSet",
        linestyle="none",
    )
    reference = n_range ** (-0.5) * n_range[0] ** (0.5) * normalized_error_mean[0]
    plt.plot(n_range, reference, label="$N^{-0.5}$", color="black", linestyle="--")
    plt.xlabel("Set size $N$")
    plt.ylabel("$|f_N(X_N) - f_{\infty}(\mu)|$")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(os.path.join(params["log_dir"], "deepset_transferability.png"))
    plt.savefig(os.path.join(params["log_dir"], "deepset_transferability.pdf"))
