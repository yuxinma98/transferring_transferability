import os
import argparse
import json
from torchmetrics import MeanSquaredError
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle

from Anydim_transferability.DeepSet.train import train
from Anydim_transferability.DeepSet.data import HausdorffDataset, HausdorffDataModule
from Anydim_transferability.DeepSet import color_dict, data_dir, plot_model_names
from Anydim_transferability import typesetting, plot_dir, str2list

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
typesetting()


def eval(model, model_name, params, test_n_range):
    """
    Evaluate the model on test set with different set sizes
    """
    model.eval()
    mse = MeanSquaredError()
    test_mse = []
    for n in test_n_range:
        dataset = HausdorffDataset(data_dir=params["data_dir"], N=1000, n=n)
        with torch.no_grad():
            y_pred = model.predict(dataset.X)
        test_mse.append(float(mse(y_pred, dataset.y)))
        if params["training_seed"] == 0:
            if n == test_n_range[0]:
                with open(
                    os.path.join(params["log_dir"], f"hausdorff_outputs_{model_name}_small.pkl"),
                    "wb",
                ) as f:
                    pickle.dump(
                        [y_pred, dataset.y],
                        f,
                    )
            if n == test_n_range[-1]:
                with open(
                    os.path.join(params["log_dir"], f"hausdorff_outputs_{model_name}_large.pkl"),
                    "wb",
                ) as f:
                    pickle.dump(
                        [y_pred, dataset.y],
                        f,
                    )
    return test_mse


def train_and_eval(params, args, model_name):
    results.setdefault(model_name, {})
    for seed in range(args.num_trials):  # run multiple trials
        if str(seed) in results[model_name]:
            continue
        params["training_seed"] = seed
        params["model"]["normalization"] = normalizations[model_name]
        data = HausdorffDataModule(
            data_dir=params["data_dir"],
            N=params["num_samples"],
            n=params["training_size"],
            batch_size=params["batch_size"],
        )
        model = train(params, data)
        test_mse = eval(model, model_name, params, args.test_n_range)
        results[model_name][str(seed)] = {
            "mse": test_mse,
        }
        with open(os.path.join(params["log_dir"], "results_hausdorff.json"), "w") as f:
            json.dump(results, f)


def plot_results(results):
    plt.figure(figsize=(6, 5))
    for i, model_name in enumerate(results.keys()):
        mse = np.array([results[model_name][str(seed)]["mse"] for seed in range(args.num_trials)])
        plt.plot(
            args.test_n_range,
            np.mean(mse, axis=0),
            "o-",
            label=plot_model_names[model_name],
            color=color_dict[model_name],
        )
        plt.fill_between(
            args.test_n_range,
            np.min(mse, axis=0),
            np.max(mse, axis=0),
            alpha=0.3,
            color=color_dict[model_name],
        )
    plt.xlabel("Test set size (n)", fontsize=16)
    plt.ylabel("Test MSE", fontsize=16)
    plt.yscale("log")
    plt.xticks(args.test_n_range)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "hausdorff_size_generalization.png"))
    plt.savefig(os.path.join(plot_dir, "hausdorff_size_generalization.pdf"))
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for i, model_name in enumerate(results.keys()):
        with open(
            os.path.join(params["log_dir"], f"hausdorff_outputs_{model_name}_small.pkl"), "rb"
        ) as f:
            y_pred, y_true = pickle.load(f)
        indices = np.random.choice(len(y_true), 500, replace=False)
        subsampled_x = y_true[indices]
        subsampled_y = y_pred[indices]
        ax[0].scatter(
            subsampled_x,
            subsampled_y,
            label=plot_model_names[model_name],
            marker="o",
            alpha=0.3,
            s=20,
            color=color_dict[model_name],
        )

        with open(
            os.path.join(params["log_dir"], f"hausdorff_outputs_{model_name}_large.pkl"), "rb"
        ) as f:
            y_pred, y_true = pickle.load(f)
        indices = np.random.choice(len(y_true), 500, replace=False)
        subsampled_x = y_true[indices]
        subsampled_y = y_pred[indices]
        ax[1].scatter(
            subsampled_x,
            subsampled_y,
            label=plot_model_names[model_name],
            marker="o",
            alpha=0.3,
            s=20,
            color=color_dict[model_name],
        )
    for i in range(2):
        ax[i].plot([0, 4], [0, 4], "--", color="black")
        ax[i].set_xlim([0, 4])
        ax[i].set_ylim([0, 4])
        ax[i].set_xlabel("True Hausdorff distance", fontsize=16)
        ax[i].set_ylabel("Predicted Hausdorff distance", fontsize=16)
        ax[i].legend(fontsize=12)
        ax[i].tick_params(axis="x", labelsize=12)
        ax[i].tick_params(axis="y", labelsize=12)
    ax[0].set_title("Test set (Set size $n=20$)", fontsize=18)
    ax[1].set_title("Test set (Set size $n=200$)", fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "hausdorff_outputs.png"))
    plt.savefig(os.path.join(plot_dir, "hausdorff_outputs.pdf"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment set-up
    parser.add_argument(
        "--test_n_range",
        type=str2list,
        default=list(np.arange(20, 210, 20)),
        help="List of test set sizes",
    )
    parser.add_argument("--training_size", type=int, default=20, help="Set size of training data")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials to run")
    parser.add_argument(
        "--num_samples", type=int, default=5000, help="Number of samples in dataset"
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
        "name": "deepset_size_generalization_hausdorff",
        "logger": True,
        "log_checkpoint": False,
        "log_model": None,
        "log_dir": os.path.join(CURRENT_DIR, "log/size_generalization"),
        # data parameters
        "data_dir": data_dir,
        "training_size": args.training_size,
        "test_n_range": args.test_n_range,
        "num_samples": args.num_samples,
        "batch_size": 128,
        # model parameters
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
        "training_seed": 42,
    }
    if not os.path.exists(params["log_dir"]):
        os.makedirs(params["log_dir"])
    if not os.path.exists(params["data_dir"]):
        os.makedirs(params["data_dir"])

    # load results
    try:
        with open(os.path.join(params["log_dir"], "results_hausdorff.json"), "r") as f:
            results = json.load(f)
    except:
        results = {}

    normalizations = {
        "DeepSet": "sum",
        "Normalized DeepSet": "mean",
        "PointNet": "max",
    }
    for models in normalizations.keys():
        train_and_eval(params, args, models)

    plot_results(results)
