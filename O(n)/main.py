import os
import argparse
from torchmetrics import MeanSquaredError, SpearmanCorrCoef
import torch
import json
import numpy as np
import matplotlib.pyplot as plt

from train import train, GWLBDataModule

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


def eval(model, params, test_n_range):
    model.eval()
    mse = MeanSquaredError().to(model.device)
    spearman = SpearmanCorrCoef().to(model.device)
    test_mse = []
    test_rank_corr = []
    for N in test_n_range:
        data = GWLBDataModule(
            fname=f"{params['data_dir']}/GWLB_points{N}_classes[2, 7].pkl",
        )
        data.prepare_data()
        with torch.no_grad():
            y_pred = model.predict(data.X_test.to(model.device))
        test_mse.append(float(mse(y_pred, data.dist_true_test.to(model.device))))
        test_rank_corr.append(
            float(
                spearman(y_pred.reshape(-1, 1), data.dist_true_test.to(model.device).reshape(-1, 1))
            )
        )
    return test_mse, test_rank_corr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment set-up
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials to run")
    # DeepSet model parameters
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=50)
    parser.add_argument("--set_channels", type=int, default=50)
    parser.add_argument("--out_channels", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=1000)

    args = parser.parse_args()

    params = {
        # logger parameters
        "project": "anydim_transferability",
        "name": "oids",
        "logger": True,
        "log_checkpoint": False,
        "log_model": None,
        "log_dir": os.path.join(CURRENT_DIR, "log/size_generalization"),
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

    results.setdefault("unnormalized", {})
    results.setdefault("normalized", {})
    results["normalized"].setdefault("mse", {})
    results["normalized"].setdefault("rank_corr", {})
    results["unnormalized"].setdefault("mse", {})
    results["unnormalized"].setdefault("rank_corr", {})
    test_n_range = [20, 100, 200, 300, 500]
    for seed in range(args.num_trials):  # run multiple trials
        if not (
            str(seed) in results["normalized"]["mse"]
            and str(seed) in results["normalized"]["rank_corr"]
        ):
            params["model"]["normalized"] = True
            model_normalized = train(params)
            mse, rank_corr = eval(model_normalized, params, test_n_range)
            results["normalized"]["mse"][str(seed)] = mse
            results["normalized"]["rank_corr"][str(seed)] = rank_corr
            with open(os.path.join(params["log_dir"], "results.json"), "w") as f:
                json.dump(results, f)
        if not (
            str(seed) in results["unnormalized"]["mse"]
            and str(seed) in results["unnormalized"]["rank_corr"]
        ):
            params["model"]["normalized"] = False
            model_normalized = train(params)
            mse, rank_corr = eval(model_normalized, params, test_n_range)
            results["unnormalized"]["mse"][str(seed)] = mse
            results["unnormalized"]["rank_corr"][str(seed)] = rank_corr
            with open(os.path.join(params["log_dir"], "results.json"), "w") as f:
                json.dump(results, f)

    # plot results
    log_mse_normalized_list = [
        np.log(results["normalized"]["mse"][str(seed)]) for seed in range(args.num_trials)
    ]
    log_mse_unnormalized_list = [
        np.log(results["unnormalized"]["mse"][str(seed)]) for seed in range(args.num_trials)
    ]
    rank_corr_normalized_list = [
        results["normalized"]["rank_corr"][str(seed)] for seed in range(args.num_trials)
    ]
    rank_corr_unnormalized_list = [
        results["unnormalized"]["rank_corr"][str(seed)] for seed in range(args.num_trials)
    ]
    mean_mse_normalized = np.mean(log_mse_normalized_list, axis=0)
    std_mse_normalized = np.std(log_mse_normalized_list, axis=0)
    mean_mse_unnormalized = np.mean(log_mse_unnormalized_list, axis=0)
    std_mse_unnormalized = np.std(log_mse_unnormalized_list, axis=0)
    mean_rank_corr_normalized = np.mean(rank_corr_normalized_list, axis=0)
    std_rank_corr_normalized = np.std(rank_corr_normalized_list, axis=0)
    mean_rank_corr_unnormalized = np.mean(rank_corr_unnormalized_list, axis=0)
    std_rank_corr_unnormalized = np.std(rank_corr_unnormalized_list, axis=0)

    plt.figure()
    plt.plot(test_n_range, mean_mse_normalized, label="Normalized")
    plt.fill_between(
        test_n_range,
        mean_mse_normalized - std_mse_normalized,
        mean_mse_normalized + std_mse_normalized,
        alpha=0.3,
    )
    plt.plot(test_n_range, mean_mse_unnormalized, label="Unnormalized")
    plt.fill_between(
        test_n_range,
        mean_mse_unnormalized - std_mse_unnormalized,
        mean_mse_unnormalized + std_mse_unnormalized,
        alpha=0.3,
    )
    plt.xlabel("Test set size (N)")
    plt.ylabel("log(Test MSE)")
    plt.legend()
    plt.savefig(os.path.join(params["log_dir"], f"test_MSE.png"))
    plt.close()

    plt.figure()
    plt.plot(test_n_range, mean_rank_corr_normalized, label="Normalized")
    plt.fill_between(
        test_n_range,
        mean_rank_corr_normalized - std_rank_corr_normalized,
        mean_rank_corr_normalized + std_rank_corr_normalized,
        alpha=0.3,
    )
    plt.plot(test_n_range, mean_rank_corr_unnormalized, label="Unnormalized")
    plt.fill_between(
        test_n_range,
        mean_rank_corr_unnormalized - std_rank_corr_unnormalized,
        mean_rank_corr_unnormalized + std_rank_corr_unnormalized,
        alpha=0.3,
    )
    plt.xlabel("Test set size (N)")
    plt.ylabel("Spearman Correlation")
    plt.legend()
    plt.savefig(os.path.join(params["log_dir"], f"test_rank_corr.png"))
    plt.close()
