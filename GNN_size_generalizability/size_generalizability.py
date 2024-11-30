import argparse
import os
import numpy as np
from typing import Union
import json
import matplotlib.pyplot as plt
import torch
from torchmetrics import MeanSquaredError
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl

from train import train
from data import DegreeDataset


def nrange(value: Union[str, list]) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return list(np.arange(*map(float, value.split(":"))))


def eval(model, params, test_n_range):
    model.eval()
    test_params = params.copy()
    test_params["batch_size"] = 5
    test_loss = np.zeros(len(test_n_range))
    for i, n in tqdm(enumerate(test_n_range)):
        test_params["n_nodes"] = int(n)
        trainer = pl.Trainer(devices=1, logger=False)
        model.params = test_params
        model.prepare_data()
        model.setup()
        test_loss[i] = trainer.test(model, verbose=False)[0].get("test_loss")
    return test_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment set-ups
    parser.add_argument(
        "--model", type=str, default="reduced", choices=["simple", "reduced", "unreduced", "ign"]
    )
    parser.add_argument("--training_graph_size", type=int, default=40)
    parser.add_argument(
        "--test_n_range",
        type=nrange,
        default="40:160:20",
        help="Range of test graph sizes",
    )
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials to run")

    # GNN parameters
    parser.add_argument("--num_layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--hidden_channels", type=int, default=5, help="Number of hidden channels")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum number of epochs")

    args = parser.parse_args()
    params = {
        # logger parameters
        "project": "anydim_transferability",  # wandb project name
        "name": "GNN_size_generalizability",  # wandb run name
        "logger": True,  # whether to use wandb to log
        "log_checkpoint": False,  # whether to log model checkpoint in wandb
        "log_model": None,  # whether to log gradient/parameters in wandb
        "log_dir": "log/size_generalizability",  # directory to save logs
        # data parameters
        "n_graphs": 5000,  # size of training dataset
        "n_nodes": args.training_graph_size,
        "test_n_range": args.test_n_range,  # range of test set sizes
        "feature_dim": 1,  # dimension of node features
        "data_dir": "data/",
        "val_fraction": 0.2,
        "test_fraction": 0.2,
        "batch_size": 128,
        "data_seed": 1,
        # model parameters
        "model": {
            "hidden_channels": args.hidden_channels,
            "num_layers": args.num_layers,
            "model": args.model,
        },
        # training parameters
        "lr": args.lr,
        "lr_patience": 10,
        "weight_decay": 0.1,
        "max_epochs": args.max_epochs,
        "training_seed": 42,
    }
    if not os.path.exists(params["log_dir"]):
        os.makedirs(params["log_dir"])

    # load results
    try:
        with open(os.path.join(params["log_dir"], f"results_{args.model}.json"), "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    results.setdefault(args.model, {})
    for seed in tqdm(range(args.num_trials)):
        if str(seed) not in results[args.model]:  # skip if already done
            params["training_seed"] = seed
            model = train(params)
            mse = eval(model, params, args.test_n_range)
            results[args.model][str(seed)] = mse
        with open(os.path.join(params["log_dir"], f"results_{args.model}.json"), "w") as f:
            json.dump(results, f)
    # plot results
    log_mse_list = [np.log(results[args.model][str(seed)]) for seed in range(args.num_trials)]
    mean_mse = np.mean(log_mse_list, axis=0)
    std_mse = np.std(log_mse_list, axis=0)

    plt.figure()
    x = np.arange(500, 5000, 500)
    plt.plot(x, mean_mse, label="Normalized")
    plt.fill_between(
        x,
        mean_mse - std_mse,
        mean_mse + std_mse,
        alpha=0.3,
    )
    plt.xlabel("Test set size (N)")
    plt.ylabel("log(Test MSE)")
    plt.title(f'Task {params["task_id"]}')
    plt.legend()
    plt.savefig(os.path.join(params["log_dir"], f"{args.model}.png"))
    plt.close()
