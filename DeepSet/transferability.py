import torch
import argparse
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from .train import train
from .data import SubsampledDataset
from .. import nrange
from . import color_dict

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def eval(model, n_range, X_reference, y_reference, n_samples, params):
    error_mean = np.zeros_like(n_range, dtype=float)
    error_std = np.zeros_like(n_range, dtype=float)
    for i, n in enumerate(n_range):
        subsampled_data = SubsampledDataset(X_reference, n_samples=n_samples, set_size=n)
        loader = DataLoader(subsampled_data, batch_size=params["batch_size"], shuffle=False)
        with torch.no_grad():
            out = []
            for batch in loader:
                out.append(model(batch))
        out = torch.cat(out, dim=0)
        error = torch.abs(out - y_reference)
        error_mean[i] = float(error.mean(dim=0).squeeze())
        error_std[i] = float(error.std(dim=0).squeeze())
    return error_mean, error_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment set-up
    parser.add_argument("--training_size", type=int, default=500, help="Set size of training data")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples to generate")
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
        "name": "deepset_transferability",
        "logger": True,
        "log_checkpoint": False,
        "log_model": None,
        "log_dir": os.path.join(CURRENT_DIR, "log/transferability"),
        # data parameters
        "data_dir": "/export/canton/data/yma93/anydim_transferability/deepset/",
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

    params["model"]["normalization"] = "mean"
    normalized_ds, _, _ = train(params)
    params["model"]["normalization"] = "sum"
    unnormalized_ds, _, _ = train(params)
    params["model"]["normalization"] = "max"
    pointnet, _, _ = train(params)

    models = {
        "Normalized DeepSet": normalized_ds,
        "DeepSet": unnormalized_ds,
        "PointNet": pointnet,
    }
    # Load distribution parameter Sigma of task 1
    with h5py.File(os.path.join(params["data_dir"], "task1/matrix_A.mat"), "r") as f:
        A = torch.tensor(f["A"][()], dtype=torch.float32)
        cov = A @ A.T

    # Take a reference set, and use its empirical distribution to sample X_n
    multivariate_normal = torch.distributions.MultivariateNormal(torch.zeros(2), cov)
    X_reference = multivariate_normal.sample((args.reference_set_size,)).unsqueeze(0)
    y_reference = {}
    with torch.no_grad():
        y_reference["Normalized DeepSet"] = float(normalized_ds(X_reference))
        y_reference["DeepSet"] = float(unnormalized_ds(X_reference))
        y_reference["PointNet"] = float(pointnet(X_reference))

    # subsample to get X_n for a range of n
    n_range = np.array(args.n_range).astype(int)
    plt.rcParams.update({"font.size": 18})  # Adjust font size as needed
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    for model_name, model in models.items():
        model.eval()
        error_mean, error_std = eval(
            model, n_range, X_reference, y_reference[model_name], args.n_samples, params
        )
        plt.errorbar(
            n_range,
            error_mean,
            error_std,
            capsize=10,
            markersize=10,
            elinewidth=2,
            fmt="o",
            color=color_dict[model_name],
            label=model_name,
            linestyle="none",
        )
        if model_name == "Normalized DeepSet":
            reference = n_range ** (-0.5) * n_range[0] ** (0.5) * error_mean[0]
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
