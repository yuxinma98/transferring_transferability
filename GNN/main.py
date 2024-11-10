import argparse
import wandb
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from data import SubsampledDataset
from train import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--hidden_channels", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=10)

    args = parser.parse_args()
    params = {
        # logger parameters
        "project": "anydim_transferability",
        "name": "GNN",
        "logger": False,
        "log_checkpoint": False,
        "log_model": None,
        "log_dir": "log/",
        # data parameters
        "batch_size": 20,
        # model parameters
        "model": {
            "in_channels": 1433,
            "hidden_channels": args.hidden_channels,
            "num_layers": args.num_layers,
            "out_channels": 7,  # classification into 7 classes
            "task": "classification",
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

    with torch.no_grad():
        predict_loader = DataLoader([model.data], batch_size=1, shuffle=False)
        reference_out = trainer.predict(model, predict_loader, ckpt_path="best")[0]
        log_n_range = np.arange(1, 3, 0.1)
        n_range = np.power(10, log_n_range).astype(int)
        errors_mean = np.zeros_like(n_range, dtype=float)
        errors_std = np.zeros_like(n_range, dtype=float)
        # errors_mean_full = np.zeros_like(n_range, dtype=float)
        # errors_std_full = np.zeros_like(n_range, dtype=float)
        for i, n in enumerate(n_range):
            subsampled_data = SubsampledDataset("data/", "Cora", args.n_samples, n)
            test_loader = DataLoader(
                [data for data in subsampled_data], batch_size=params["batch_size"], shuffle=False
            )
            out = trainer.predict(model, test_loader, ckpt_path="best")
            out = torch.cat(out, dim=0)
            errors = torch.abs(out - reference_out)
            errors_mean[i] = errors.mean().item()
            errors_std[i] = errors.std().item()

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
    plt.savefig(os.path.join(params["log_dir"], f"transferability.png"))
    if params["logger"]:
        wandb.log({"transferability": wandb.Image(os.path.join(params["log_dir"], f"transferability.png"))})
        wandb.finish()
