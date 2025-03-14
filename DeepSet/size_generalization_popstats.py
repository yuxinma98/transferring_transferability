import os
import argparse
import json
from torchmetrics import MeanSquaredError
import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle

from Anydim_transferability.DeepSet.train import train
from Anydim_transferability.DeepSet.data import PopStatsDataModule, PopStatsDataset
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
    for N in test_n_range:
        dataset = PopStatsDataset(
            fname=os.path.join(params["data_dir"], f'task{params["task_id"]}/test_{N}.mat')
        )
        with torch.no_grad():
            y_pred = model.predict(dataset.X)
        test_mse.append(float(mse(y_pred, dataset.y)))
        if params["training_seed"]:
            if N == test_n_range[0]:
                with open(
                    os.path.join(
                        params["log_dir"],
                        f"popstats_outputs_{params['task_id']}_{model_name}_small.pkl",
                    ),
                    "wb",
                ) as f:
                    pickle.dump(
                        [dataset.t, y_pred.squeeze()],
                        f,
                    )
            if N == test_n_range[-1]:
                with open(
                    os.path.join(
                        params["log_dir"],
                        f"popstats_outputs_{params['task_id']}_{model_name}_large.pkl",
                    ),
                    "wb",
                ) as f:
                    pickle.dump(
                        [dataset.t, y_pred.squeeze()],
                        f,
                    )
    return test_mse


def train_and_eval(params, args, model_name):
    for task_id in [1, 2, 3, 4]:
        params["task_id"] = task_id
        results.setdefault(f"task{task_id}", {}).setdefault(model_name, {})
        for seed in range(args.num_trials):  # run multiple trials
            if str(seed) in results[f"task{task_id}"][model_name]:
                continue
            params["training_seed"] = seed
            params["model"]["normalization"] = normalizations[model_name]
            data = PopStatsDataModule(
                data_dir=params["data_dir"],
                task_id=params["task_id"],
                batch_size=params["batch_size"],
                training_size=params["training_size"],
            )
            model = train(params, data)
            test_mse = eval(model, model_name, params, args.test_n_range)
            results[f"task{task_id}"][model_name][str(seed)] = {
                "mse": test_mse,
            }
            with open(os.path.join(params["log_dir"], "results_popstats.json"), "w") as f:
                json.dump(results, f)


def plot_results(results):
    fig, axes = plt.subplots(3, 4, figsize=(20, 13))
    titles = {
        1: "Task1: Rotation",
        2: "Task2: Correlation",
        3: "Task3: Rank 1",
        4: "Task4: Random",
    }
    ylabels = {
        1: "Entropy",
        2: "Mutual Information",
        3: "Mutual Information",
        4: "Mutual information",
    }
    xlabels = {1: "Rotation Angle", 2: "Correlation", 3: "Rank-1 Length", 4: "Sorted Index"}
    subsample_size = 300
    for task_id in [1, 2, 3, 4]:
        # plot MSE
        ax = axes[0, task_id - 1]
        for model_name in normalizations.keys():
            mse_list = [
                results[f"task{task_id}"][model_name][str(seed)]["mse"]
                for seed in range(args.num_trials)
            ]
            mean_mse = np.mean(mse_list, axis=0)
            ax.plot(
                args.test_n_range,
                mean_mse,
                "o-",
                label=plot_model_names[model_name],
                color=color_dict[model_name],
            )
            ax.fill_between(
                args.test_n_range,
                np.min(mse_list, axis=0),
                np.max(mse_list, axis=0),
                alpha=0.3,
                color=color_dict[model_name],
            )
        ax.set_yscale("log")
        ax.set_xlabel("Test set size (n)", fontsize=16)
        ax.set_ylabel("Test MSE", fontsize=16)
        ax.set_xticks(args.test_n_range)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.legend(fontsize=12, loc="upper right")
        ax.set_title(f"{titles[task_id]}", fontsize=18)

        # plot predictions
        data = PopStatsDataModule(
            data_dir=params["data_dir"],
            task_id=task_id,
            batch_size=params["batch_size"],
            training_size=params["training_size"],
        )
        data.setup()
        truth = data.truth
        ax = axes[1, task_id - 1]
        for model_name in normalizations.keys():
            with open(
                os.path.join(
                    params["log_dir"],
                    f"popstats_outputs_{task_id}_{model_name}_small.pkl",
                ),
                "rb",
            ) as f:
                t, test_out = pickle.load(f)
            indices = np.random.choice(
                test_out.shape[0], subsample_size, replace=False
            )  # subsample for visualization
            subsampled_x = t[indices]
            subsampled_y = test_out[indices]
            ax.scatter(
                subsampled_x,
                subsampled_y,
                marker="o",
                alpha=0.3,
                s=20,
                label=plot_model_names[model_name],
                color=color_dict[model_name],
            )

        ax.plot(truth.t, truth.y, label="Truth", color="black", linestyle="--", linewidth=2)
        ax.set_xlabel(xlabels[task_id], fontsize=16)
        ax.set_ylabel(ylabels[task_id], fontsize=16)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.legend(loc="upper right", fontsize=12)

        # plot predictions for large n
        ax = axes[2, task_id - 1]
        for model_name in normalizations.keys():
            with open(
                os.path.join(
                    params["log_dir"],
                    f"popstats_outputs_{task_id}_{model_name}_large.pkl",
                ),
                "rb",
            ) as f:
                t, test_out = pickle.load(f)
            indices = np.random.choice(
                test_out.shape[0], subsample_size, replace=False
            )  # subsample for visualization
            subsampled_x = t[indices]
            subsampled_y = test_out[indices]
            ax.scatter(
                subsampled_x,
                subsampled_y,
                marker="o",
                alpha=0.3,
                s=20,
                label=plot_model_names[model_name],
                color=color_dict[model_name],
            )

        ax.plot(truth.t, truth.y, label="Truth", color="black", linestyle="--", linewidth=2)
        ax.set_xlabel(xlabels[task_id], fontsize=16)
        ax.set_ylabel(ylabels[task_id], fontsize=16)
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.legend(loc="upper right", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "deepset_plot.pdf"))
    plt.savefig(os.path.join(plot_dir, "deepset_plot.png"))
    plt.close()


def plot_task_3_mse(results):
    task_id = 3
    plt.figure(figsize=(6, 5))

    for model_name in normalizations.keys():
        mse_list = [
            results[f"task{task_id}"][model_name][str(seed)]["mse"]
            for seed in range(args.num_trials)
        ]
        mean_mse = np.mean(mse_list, axis=0)
        plt.plot(
            args.test_n_range,
            mean_mse,
            "o-",
            label=plot_model_names[model_name],
            color=color_dict[model_name],
        )
        plt.fill_between(
            args.test_n_range,
            np.min(mse_list, axis=0),
            np.max(mse_list, axis=0),
            alpha=0.3,
            color=color_dict[model_name],
        )

    plt.yscale("log")
    plt.xlabel("Test set size (n)", fontsize=16)
    plt.ylabel("Test MSE", fontsize=16)
    plt.xticks(args.test_n_range)
    plt.tick_params(axis="x", labelsize=12)
    plt.tick_params(axis="y", labelsize=12)
    plt.legend(fontsize=12, loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "task_3_mse_plot.pdf"))
    plt.savefig(os.path.join(plot_dir, "task_3_mse_plot.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment set-up
    parser.add_argument(
        "--test_n_range",
        type=str2list,
        default=list(np.arange(500, 5000, 500)),
        help="List of test set sizes",
    )
    parser.add_argument("--training_size", type=int, default=500, help="Set size of training data")
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials to run")

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
        "name": "deepset_size_generalization_popstats",
        "logger": True,
        "log_checkpoint": False,
        "log_model": None,
        "log_dir": os.path.join(CURRENT_DIR, "log/size_generalization"),
        # data parameters
        "data_dir": data_dir,
        "training_size": args.training_size,
        "test_n_range": args.test_n_range,
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
        with open(os.path.join(params["log_dir"], "results_popstats.json"), "r") as f:
            results = json.load(f)
    except:
        results = {}

    normalizations = {
        "DeepSet": "sum",
        "Normalized DeepSet": "mean",
        "PointNet": "max",
    }
    for model_name in normalizations.keys():
        train_and_eval(params, args, model_name)

    plot_results(results)
    plot_task_3_mse(results)
