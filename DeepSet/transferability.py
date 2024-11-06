import os
import argparse
import json
from torchmetrics import MeanSquaredError
import matplotlib.pyplot as plt
import numpy as np
from data import PopStatsDataset
from train import train

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def eval(model, params):
    model.eval()
    mse = MeanSquaredError()
    N = params["testing_size"]
    dataset = PopStatsDataset(fname = os.path.join(params["data_dir"], f'task{params["task_id"]}/test_{N}.mat'))
    y_pred = model.predict(dataset.X)
    return float(mse(y_pred, dataset.y))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--testing_size", type=int, default=5000)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=50)
    parser.add_argument("--set_channels", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=1000)

    args = parser.parse_args()

    params = {
        # logger parameters
        "project": "anydim_transferability",
        "name": "transferability",
        "logger": True,
        "log_checkpoint": False,
        "log_model": None,
        "log_dir": os.path.join(CURRENT_DIR, "log/transferability"),
        # data parameters
        "data_dir": os.path.join(CURRENT_DIR, "generator/data"),
        "batch_size": 128,
        "testing_size": args.testing_size,
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

    for task_id in [1,2,3,4]:
        params["task_id"] = task_id
        results.setdefault(str(task_id), {}).setdefault("normalized", {})
        results[str(task_id)].setdefault("unnormalized", {})

        # run experiments
        for seed in range(args.num_trials):
            if str(seed) not in results[str(task_id)]["normalized"]:
                params["model"]["normalized"] = True
                mse_normalized = []
                params["training_seed"] = seed
                for training_size in range(500, 3000, 500):
                    params["training_size"] = training_size
                    model_normalized = train(params)
                    mse_normalized.append(eval(model_normalized, params))
                results[str(task_id)]["normalized"][str(seed)] = mse_normalized

            if str(seed) not in results[str(task_id)]["unnormalized"]:
                params["model"]["normalized"] = False
                mse_unnormalized = []
                for training_size in range(500, 3000, 500):
                    params["training_size"] = training_size
                    model_unnormalized = train(params)
                    mse_unnormalized.append(eval(model_unnormalized, params))
                results[str(task_id)]["unnormalized"][str(seed)] = mse_unnormalized
            with open(os.path.join(params["log_dir"], "results.json"), "w") as f:
                json.dump(results, f)

        mse_normalized_list = [
            results[str(task_id)]["normalized"][str(seed)] for seed in range(args.num_trials)
        ]
        mse_unnormalized_list = [
            results[str(task_id)]["unnormalized"][str(seed)] for seed in range(args.num_trials)
        ]
        mean_mse_normalized = np.mean(mse_normalized_list, axis=0)
        mean_mse_unnormalized = np.mean(mse_unnormalized_list, axis=0)
        lower_quantile_normalized = np.quantile(mse_normalized_list, q=0.25, axis=0)
        lower_quantile_unnormalized = np.quantile(mse_unnormalized_list, q=0.25, axis=0)
        upper_quantile_normalized = np.quantile(mse_normalized_list, q=0.75, axis=0)
        upper_quantile_unnormalized = np.quantile(mse_unnormalized_list, q=0.75, axis=0)

        # plot results
        plt.plot(np.arange(500, 3000, 500), mean_mse_normalized, label="Normalized")
        plt.fill_between(
            np.arange(500, 3000, 500),
            lower_quantile_normalized,
            upper_quantile_normalized,
            alpha=0.3,
        )
        plt.plot(np.arange(500, 3000, 500), mean_mse_unnormalized, label="Unnormalized")
        plt.fill_between(
            np.arange(500, 3000, 500),
            lower_quantile_unnormalized,
            upper_quantile_unnormalized,
            alpha=0.3,
        )
        plt.xlabel('Train set size (N)')
        plt.ylabel(f'Test MSE on N = {params["testing_size"]}')
        plt.title(f'Task {params["task_id"]}')
        plt.legend()
        plt.savefig(os.path.join(params["log_dir"], f'task{params["task_id"]}_plot.png'))
