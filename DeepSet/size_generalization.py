import os
import argparse
import json
from torchmetrics import MeanSquaredError
import matplotlib.pyplot as plt
import numpy as np
from data import PopStatsDataset

from train import train
from data import PopStatsDataModule

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ibm_colors = [
    "#648FFF",  # Blue
    "#785EF0",  # Purple
    "#DC267F",  # Pink
    "#FE6100",  # Orange
    "#FFB000",  # Yellow
]


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def eval(model, params, test_n_range):
    model.eval()
    mse = MeanSquaredError()
    test_mse = []
    for N in test_n_range:
        dataset = PopStatsDataset(
            fname=os.path.join(params["data_dir"], f'task{params["task_id"]}/test_{N}.mat')
        )
        y_pred = model.predict(dataset.X)
        test_mse.append(float(mse(y_pred, dataset.y)))
    return test_mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment set-up
    parser.add_argument(
        "--test_n_range",
        type=list,
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
        "name": "deepset_size_generalization",
        "logger": True,
        "log_checkpoint": False,
        "log_model": None,
        "log_dir": os.path.join(CURRENT_DIR, "log/size_generalization"),
        # data parameters
        "data_dir": "/export/canton/data/yma93/anydim_transferability/deepset/",
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
        with open(os.path.join(params["log_dir"], "results.json"), "r") as f:
            results = json.load(f)
    except:
        results = {}

    fig, axes = plt.subplots(3, 4, figsize=(20, 10))
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

    for task_id in [1, 2, 3, 4]:
        params["task_id"] = task_id
        results.setdefault(f"task{task_id}", {}).setdefault("normalized", {})
        results[f"task{task_id}"].setdefault("unnormalized", {})

        # run experiments
        for seed in range(args.num_trials):  # run multiple trials
            params["training_seed"] = seed
            if str(seed) not in results[f"task{task_id}"]["normalized"]:  # skip if already done
                params["model"]["normalized"] = True
                model_normalized, test_out, test_out_large_n = train(params)
                mse_normalized = eval(model_normalized, params, args.test_n_range)
                results[f"task{task_id}"]["normalized"][str(seed)] = {
                    "mse": mse_normalized,
                }
                if seed == 0:
                    results[f"task{task_id}"]["normalized"]["test_out"] = test_out
                    results[f"task{task_id}"]["normalized"]["test_out_large_n"] = test_out_large_n

            if str(seed) not in results[f"task{task_id}"]["unnormalized"]:
                params["model"]["normalized"] = False
                model_unnormalized, test_out, test_out_large_n = train(params)
                mse_unnormalized = eval(model_unnormalized, params, args.test_n_range)
                results[f"task{task_id}"]["unnormalized"][str(seed)] = {"mse": mse_unnormalized}
                if seed == 0:
                    results[f"task{task_id}"]["unnormalized"]["test_out"] = test_out
                    results[f"task{task_id}"]["unnormalized"]["test_out_large_n"] = test_out_large_n
            with open(os.path.join(params["log_dir"], "results.json"), "w") as f:
                json.dump(results, f)

        # plot MSE
        log_mse_normalized_list = [
            np.log(results[f"task{task_id}"]["normalized"][str(seed)]["mse"])
            for seed in range(args.num_trials)
        ]
        log_mse_unnormalized_list = [
            np.log(results[f"task{task_id}"]["unnormalized"][str(seed)]["mse"])
            for seed in range(args.num_trials)
        ]
        mean_mse_normalized = np.mean(log_mse_normalized_list, axis=0)
        mean_mse_unnormalized = np.mean(log_mse_unnormalized_list, axis=0)

        ax = axes[0, task_id - 1]
        ax.plot(
            args.test_n_range,
            mean_mse_normalized,
            "o-",
            label="Normalized DeepSet",
            color=ibm_colors[0],
        )
        ax.fill_between(
            args.test_n_range,
            np.min(log_mse_normalized_list, axis=0),
            np.max(log_mse_normalized_list, axis=0),
            alpha=0.3,
            color=ibm_colors[0],
        )
        ax.plot(
            args.test_n_range, mean_mse_unnormalized, "o-", label="DeepSet", color=ibm_colors[3]
        )
        ax.fill_between(
            args.test_n_range,
            np.min(log_mse_unnormalized_list, axis=0),
            np.max(log_mse_unnormalized_list, axis=0),
            alpha=0.3,
            color=ibm_colors[3],
        )
        ax.set_xlabel("Test set size (N)", fontsize=18)
        ax.set_ylabel("log(Test MSE)", fontsize=18)
        ax.set_xticks(args.test_n_range)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=14)
        ax.legend(fontsize=16, loc="upper right")
        ax.set_title(f"{titles[task_id]}", fontsize=18)

        # plot output
        data = PopStatsDataModule(
            data_dir=params["data_dir"],
            task_id=params["task_id"],
            batch_size=params["batch_size"],
            training_size=params["training_size"],
        )
        data.setup()
        truth = data.truth
        test_out_normalized = results[f"task{task_id}"]["normalized"]["test_out"]
        test_out_unnormalized = results[f"task{task_id}"]["unnormalized"]["test_out"]

        ax = axes[1, task_id - 1]
        ax.plot(
            test_out_unnormalized[0],
            test_out_unnormalized[1],
            "x",
            label="DeepSet",
            color=ibm_colors[3],
        )
        ax.plot(
            test_out_normalized[0],
            test_out_normalized[1],
            "x",
            label="Normalized DeepSet",
            color=ibm_colors[0],
        )
        ax.plot(truth.t, truth.y, label="truth", color="black", linestyle="--", linewidth=3)
        ax.set_xlabel(xlabels[task_id], fontsize=18)
        ax.set_ylabel(ylabels[task_id], fontsize=18)
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.legend(loc="upper right", fontsize=16)

        # plot output for large n
        truth = data.truth
        test_out_normalized = results[f"task{task_id}"]["normalized"]["test_out_large_n"]
        test_out_unnormalized = results[f"task{task_id}"]["unnormalized"]["test_out_large_n"]

        ax = axes[2, task_id - 1]

        ax.plot(
            test_out_unnormalized[0],
            test_out_unnormalized[1],
            "x",
            label="DeepSet",
            color=ibm_colors[3],
        )
        ax.plot(
            test_out_normalized[0],
            test_out_normalized[1],
            "x",
            label="Normalized DeepSet",
            color=ibm_colors[0],
        )
        ax.plot(truth.t, truth.y, label="truth", color="black", linestyle="--", linewidth=3)
        ax.set_xlabel(xlabels[task_id], fontsize=18)
        ax.set_ylabel(ylabels[task_id], fontsize=18)
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        ax.legend(loc="upper right", fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(params["log_dir"], "deepset_plot.pdf"))
    plt.savefig(os.path.join(params["log_dir"], "deepset_plot.png"))
    plt.close()
