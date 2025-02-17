import argparse
import os
import numpy as np
from typing import Union
import json
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from torchmetrics import MeanSquaredError
from tqdm import tqdm
import pickle, os

from Anydim_transferability.GNN.train import train
from Anydim_transferability.GNN.data import HomDensityDataset
from Anydim_transferability.GNN import data_dir, color_dict, plot_model_names
from Anydim_transferability import typesetting, nrange, plot_dir, str2list

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
typesetting()


def eval(model, params, test_n_range, record_out=False):
    model.eval()
    test_params = params.copy()
    test_params["batch_size"] = 5
    test_loss = np.zeros(len(test_n_range))
    mean_squared_error = MeanSquaredError()
    outputs = []
    truths = []
    for i, n in tqdm(enumerate(test_n_range)):
        test_params["n_nodes"] = int(n)
        test_dataset = HomDensityDataset(
            root=params["data_dir"],
            N=1000,
            n=test_params["n_nodes"],
            graph_model=params["graph_model"],
            task=params["task"],
        )
        test_loader = DataLoader(test_dataset, batch_size=test_params["batch_size"])

        with torch.no_grad():
            for batch in test_loader:
                out = model(batch)
                # batch.y[batch.y == 0] = 1e-9  # avoid division by zero
                # relative_error = abs(out.reshape(-1) - batch.y) / abs(
                #     batch.y
                # )  # dim: (batch_size * n)
                # test_loss[i] += relative_error.sum().item()
                test_loss[i] += mean_squared_error(out.reshape(-1), batch.y).item() * len(batch.y)
                if record_out and n == test_n_range[-1]:
                    outputs.extend(out.reshape(-1).tolist())
                    truths.extend(batch.y.tolist())
        test_loss[i] /= len(test_dataset) * n
    if record_out:
        with open(
            os.path.join(
                params["log_dir"],
                f"outputs_{params['graph_model']}_{params['task']}_{params['model']['model']}_largetest.pkl",
            ),
            "wb",
        ) as f:
            pickle.dump({"outputs": outputs, "truths": truths}, f)
    return test_loss.tolist()


def train_and_eval(params, args):
    results.setdefault(params["model"]["model"], {})
    seed = 0
    for trial in range(args.num_trials):  # run multiple trials
        if str(trial) in results[params["model"]["model"]]:
            continue
        best_model = None
        for i in range(5):  # for each trial, take the best out of 5 random runs
            seed = trial * 5 + i
            params["training_seed"] = seed
            model, val_loss = train(params)
            if best_model is None or val_loss < best_loss:
                best_model, best_loss = (model, val_loss)
                if trial == 0:  # only record outputs for the first trial
                    # generate output of train dataset and test dataset
                    model.eval()
                    with torch.no_grad():
                        train_outputs, train_truth = [], []
                        test_outputs, test_truth = [], []
                        for data in model.train_dataloader():
                            out = model(data)
                            train_outputs.extend(out.reshape(-1).tolist())
                            train_truth.extend(data.y.tolist())
                        for data in model.test_dataloader():
                            out = model(data)
                            test_outputs.extend(out.reshape(-1).tolist())
                            test_truth.extend(data.y.tolist())
                    # save outputs and truth to file
                    with open(
                        os.path.join(
                            params["log_dir"],
                            f"outputs_{params['graph_model']}_{params['task']}_{params['model']['model']}.pkl",
                        ),
                        "wb",
                    ) as f:
                        pickle.dump(
                            {
                                "train_outputs": train_outputs,
                                "train_truths": train_truth,
                                "test_outputs": test_outputs,
                                "test_truths": test_truth,
                            },
                            f,
                        )
        mse = eval(
            best_model, params, test_n_range, record_out=(trial == 0)
        )  # evaluate the best model on a range of test sizes
        results[params["model"]["model"]][str(trial)] = mse
        with open(os.path.join(params["log_dir"], fname), "w") as f:
            json.dump(results, f)


def plot_size_generalization():
    plt.figure(figsize=(8, 5))
    fig, ax = plt.subplots()
    for model_name in model_params.keys():
        model = model_params[model_name]["model"]
        mse_list = [results[model][str(seed)] for seed in range(args.num_trials)]
        ax.plot(
            np.array(test_n_range),
            np.median(mse_list, axis=0),
            label=plot_model_names[model_name],
            color=color_dict[model_name],
        )
        ax.fill_between(
            np.array(test_n_range),
            np.percentile(mse_list, 20, axis=0),
            np.percentile(mse_list, 80, axis=0),
            alpha=0.3,
            color=color_dict[model_name],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Test graph size (M)", fontsize=16)
    ax.set_ylabel("Test MSE", fontsize=16)
    ax.legend(loc="upper right", fontsize=12)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            plot_dir,
            f"gnn_{params['graph_model']}_{params['task']}.png",
        )
    )
    plt.savefig(
        os.path.join(
            plot_dir,
            f"gnn_{params['graph_model']}_{params['task']}.pdf",
        )
    )
    plt.close()


def plot_output():
    fig, axs = plt.subplots(1, 3, figsize=(14, 5))
    for model_name in model_params.keys():
        model = model_params[model_name]["model"]
        fname = f"outputs_full_SBM_Gaussian_triangle_{model}.pkl"
        large_fname = f"outputs_full_SBM_Gaussian_triangle_{model}_largetest.pkl"
        with open(os.path.join(params["log_dir"], fname), "rb") as f:
            out = pickle.load(f)
        with open(os.path.join(params["log_dir"], large_fname), "rb") as f:
            out_large = pickle.load(f)

        # Training data
        train_subset = np.random.choice(len(out["train_outputs"]), 500, replace=False)
        train_subset = train_subset.astype(int)
        axs[0].scatter(
            np.array(out["train_truths"])[train_subset],
            np.array(out["train_outputs"])[train_subset],
            label=plot_model_names[model_name],
            alpha=0.3,
            s=20,
            marker="o",
            color=color_dict[model_name],
        )
        axs[0].set_title(r"Training Set (Graph size $M=50$)", fontsize=18)

        # Test data
        test_subset = np.random.choice(len(out["test_outputs"]), 500, replace=False)
        test_subset = test_subset.astype(int)
        axs[1].scatter(
            np.array(out["test_truths"])[test_subset],
            np.array(out["test_outputs"])[test_subset],
            label=plot_model_names[model_name],
            alpha=0.3,
            s=20,
            marker="o",
            color=color_dict[model_name],
        )
        axs[1].set_title(r"Test Set (Graph size $M=50$)", fontsize=18)

        # Large test data
        large_test_subset = np.random.choice(len(out_large["outputs"]), 500, replace=False)
        large_test_subset = large_test_subset.astype(int)
        axs[2].scatter(
            np.array(out_large["truths"])[large_test_subset],
            np.array(out_large["outputs"])[large_test_subset],
            label=plot_model_names[model_name],
            alpha=0.3,
            s=20,
            marker="o",
            color=color_dict[model_name],
        )
        axs[2].set_title(r"Test Set (Graph size $M\approx 2000$)", fontsize=18)
    for ax in axs:
        ax.plot([0, 4], [0, 4], "--", color="black")
        ax.set_xlim([0, 4])
        ax.set_ylim([0, 4])
        ax.set_aspect("equal")
        ax.set_xlabel("Target", fontsize=15)
        ax.set_ylabel("Prediction", fontsize=15)
        ax.legend(fontsize=12)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            plot_dir,
            f"gnn_outputs_{params['graph_model']}_{params['task']}.png",
        )
    )
    plt.savefig(
        os.path.join(
            plot_dir,
            f"gnn_outputs_{params['graph_model']}_{params['task']}.pdf",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment set-ups
    parser.add_argument(
        "--graph_model",
        type=str,
        default="full_random",
        choices=["SBM", "full_random"],
    )
    parser.add_argument(
        "--task",
        type=str,
        default="triangle",
        choices=["degree", "triangle"],
    )
    parser.add_argument("--training_graph_size", type=int, default=50)
    parser.add_argument(
        "--test_n_range",
        type=str2list,
        default=[50, 200, 500, 1000, 2000],
        help="List of test set sizes",
    )
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials to run")

    # GNN parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum number of epochs")

    args = parser.parse_args()
    params = {
        # logger parameters
        "project": "anydim_transferability",  # wandb project name
        "name": "GNN_size_generalizability",  # wandb run name
        "logger": True,  # whether to use wandb to log
        "log_checkpoint": True,  # whether to log model checkpoint in wandb
        "log_model": None,  # whether to log gradient/parameters in wandb
        "log_dir": os.path.join(CURRENT_DIR, "log/size_generalization"),  # directory to save logs
        # data parameters
        "n_graphs": 5000,  # size of training dataset
        "n_nodes": args.training_graph_size,
        "graph_model": args.graph_model,
        "task": args.task,
        "data_dir": data_dir,
        "val_fraction": 0.2,
        "test_fraction": 0.2,
        "batch_size": 128,
        "data_seed": 1,
        # training parameters
        "lr": args.lr,
        "lr_patience": 10,
        "weight_decay": 0.1,
        "max_epochs": args.max_epochs,
    }
    if not os.path.exists(params["log_dir"]):
        os.makedirs(params["log_dir"])
    test_n_range = np.array(args.test_n_range, dtype=int)

    # load results
    fname = f"results_{args.graph_model}_{args.task}.json"
    try:
        with open(os.path.join(params["log_dir"], fname), "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        results = {}

    model_params = {
        "IGN": {"model": "ign", "channel_list": [2, 6, 7, 6, 6, 1], "bias": True},
        "GNN": {"model": "simple", "channel_list": [1, 18, 18, 18, 18, 1], "bias": True},
        "GGNN": {
            "model": "unreduced",
            "channel_list": [1, 5, 5, 5, 4, 1],
            "bias": True,
        },
        "Continuous GGNN": {
            "model": "reduced",
            "channel_list": [1, 5, 6, 6, 4, 1],
            "bias": True,
        },
    }
    for model_name in model_params.keys():
        params["model"] = model_params[model_name]
        train_and_eval(params, args)

    plot_size_generalization()
    plot_output()
