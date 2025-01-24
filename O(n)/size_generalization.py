import os
import argparse
from torchmetrics import MeanSquaredError, SpearmanCorrCoef
import torch
import json
import numpy as np
import matplotlib.pyplot as plt

from train import train, GWLBDataModule

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
    """
    Evaluate the trained model on test set of different sizes
    """
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


def plot_size_generalization(results, params):
    # plot results
    log_mse_normalized_list = [
        np.log(results["normalized"][str(seed)]["mse"]) for seed in range(args.num_trials)
    ]
    log_mse_unnormalized_list = [
        np.log(results["unnormalized"][str(seed)]["mse"]) for seed in range(args.num_trials)
    ]
    rank_corr_normalized_list = [
        results["normalized"][str(seed)]["rank_corr"] for seed in range(args.num_trials)
    ]
    rank_corr_unnormalized_list = [
        results["unnormalized"][str(seed)]["rank_corr"] for seed in range(args.num_trials)
    ]

    plt.figure(figsize=(14, 5))

    # First subplot for Test MSE
    plt.subplot(1, 2, 1)
    plt.plot(
        test_n_range,
        np.median(log_mse_normalized_list, axis=0),
        "o-",
        label="Normalized SVD-DeepSet",
        color=ibm_colors[0],
    )
    plt.fill_between(
        test_n_range,
        np.median(log_mse_normalized_list, axis=0) - np.std(log_mse_normalized_list, axis=0),
        np.median(log_mse_normalized_list, axis=0) + np.std(log_mse_normalized_list, axis=0),
        # np.min(log_mse_normalized_list, axis=0),
        # np.max(log_mse_normalized_list, axis=0),
        alpha=0.3,
        color=ibm_colors[0],
    )
    plt.plot(
        test_n_range,
        np.median(log_mse_unnormalized_list, axis=0),
        "o-",
        label="SVD-DeepSet",
        color=ibm_colors[3],
    )
    plt.fill_between(
        test_n_range,
        np.median(log_mse_unnormalized_list, axis=0) - np.std(log_mse_unnormalized_list, axis=0),
        np.median(log_mse_unnormalized_list, axis=0) + np.std(log_mse_unnormalized_list, axis=0),
        # np.min(log_mse_unnormalized_list, axis=0),
        # np.max(log_mse_unnormalized_list, axis=0),
        alpha=0.3,
        color=ibm_colors[3],
    )
    plt.xlabel("Test point cloud sizes (N)", fontsize=18)
    plt.ylabel("log(Test MSE)", fontsize=18)
    plt.xticks(test_n_range, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)

    # Second subplot for Spearman Correlation
    plt.subplot(1, 2, 2)
    plt.plot(
        test_n_range,
        np.median(rank_corr_normalized_list, axis=0),
        "o-",
        label="Normalized SVD-DeepSet",
        color=ibm_colors[0],
    )
    plt.fill_between(
        test_n_range,
        np.mean(rank_corr_normalized_list, axis=0) - np.std(rank_corr_normalized_list, axis=0),
        np.mean(rank_corr_normalized_list, axis=0) + np.std(rank_corr_normalized_list, axis=0),
        # np.min(rank_corr_normalized_list, axis=0),
        # np.max(rank_corr_normalized_list, axis=0),
        alpha=0.3,
        color=ibm_colors[0],
    )
    plt.plot(
        test_n_range,
        np.median(rank_corr_unnormalized_list, axis=0),
        "o-",
        label="SVD-DeepSet",
        color=ibm_colors[3],
    )
    plt.fill_between(
        test_n_range,
        np.mean(rank_corr_unnormalized_list, axis=0) - np.std(rank_corr_unnormalized_list, axis=0),
        np.mean(rank_corr_unnormalized_list, axis=0) + np.std(rank_corr_unnormalized_list, axis=0),
        # np.min(rank_corr_unnormalized_list, axis=0),
        # np.max(rank_corr_unnormalized_list, axis=0),
        alpha=0.3,
        color=ibm_colors[3],
    )
    plt.xlabel("Test point cloud sizes (N)", fontsize=18)
    plt.ylabel("Rank Correlation", fontsize=18)
    plt.xticks(test_n_range, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(params["log_dir"], "SVD-DS_plot.png"))
    plt.savefig(os.path.join(params["log_dir"], "SVD-DS_plot.pdf"))
    plt.close()


def plot_output(results, params):
    data = GWLBDataModule(
        fname=f"{params['data_dir']}/GWLB_points{params['point_cloud_size']}_classes[2, 7].pkl",
    )
    data.prepare_data()

    plt.figure(figsize=(14, 5))

    # First subplot for Training Set
    plt.subplot(1, 3, 1)
    plt.scatter(
        data.dist_true_train,
        results["normalized"]["train_out"],
        label="Normalized SVD-DeepSet",
        alpha=0.75,
    )
    plt.scatter(
        data.dist_true_train, results["unnormalized"]["train_out"], label="SVD-DeepSet", alpha=0.75
    )
    plt.plot(np.linspace(0, 1.8, 100), np.linspace(0, 1.8, 100), ls=":", color="gray", alpha=0.5)
    plt.xlabel("Target", fontsize=15)
    plt.ylabel("Prediction", fontsize=15)
    plt.title("Training Set ($N=20$)", fontsize=18)
    plt.legend(fontsize=12)
    plt.xlim(0, 1.8)
    plt.ylim(0, 1.8)
    plt.gca().set_aspect("equal")

    # Second subplot for Test Set
    plt.subplot(1, 3, 2)
    plt.scatter(
        data.dist_true_test,
        results["normalized"]["test_out"],
        label="Normalized SVD-DeepSet",
        alpha=0.75,
    )
    plt.scatter(
        data.dist_true_test, results["unnormalized"]["test_out"], label="SVD-DeepSet", alpha=0.75
    )
    plt.plot(np.linspace(0, 1.8, 100), np.linspace(0, 1.8, 100), ls=":", color="gray", alpha=0.5)
    plt.xlabel("Target", fontsize=15)
    plt.ylabel("Prediction", fontsize=15)
    plt.title("Test Set ($N=20$)", fontsize=18)
    plt.legend(fontsize=12)
    plt.xlim(0, 1.8)
    plt.ylim(0, 1.8)
    plt.gca().set_aspect("equal")

    # Third subplot for Test Set with large n
    plt.subplot(1, 3, 3)
    plt.scatter(
        data.dist_true_test,
        results["normalized"]["test_out_large_n"],
        label="Normalized SVD-DeepSet",
        alpha=0.75,
    )
    plt.scatter(
        data.dist_true_test,
        results["unnormalized"]["test_out_large_n"],
        label="SVD-DeepSet",
        alpha=0.75,
    )
    plt.plot(np.linspace(0, 1.8, 100), np.linspace(0, 1.8, 100), ls=":", color="gray", alpha=0.5)
    plt.xlabel("Target", fontsize=15)
    plt.ylabel("Prediction", fontsize=15)
    plt.title("Test Set ($N=500$)", fontsize=18)
    plt.legend(fontsize=12)
    plt.xlim(0, 1.8)
    plt.ylim(0, 1.8)
    plt.gca().set_aspect("equal")

    plt.tight_layout()
    plt.savefig(os.path.join(params["log_dir"], "SVD-DS_output_plot.png"))
    plt.savefig(os.path.join(params["log_dir"], "SVD-DS_output_plot.pdf"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment set-up
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials to run")
    # DeepSet model parameters
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=10)
    parser.add_argument("--set_channels", type=int, default=10)
    parser.add_argument("--out_channels", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--max_epochs", type=int, default=5000)

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
    test_n_range = [20, 100, 200, 300, 500]
    for seed in range(args.num_trials):  # run multiple trials
        params["training_seed"] = seed
        if not (str(seed) in results["normalized"]):
            params["model"]["normalized"] = True
            model_normalized, train_out, test_out, test_out_large_n = train(params)
            mse, rank_corr = eval(model_normalized, params, test_n_range)
            results["normalized"][str(seed)] = {"mse": mse, "rank_corr": rank_corr}
            if seed == 0:
                results["normalized"]["train_out"] = train_out.squeeze().tolist()
                results["normalized"]["test_out"] = test_out.squeeze().tolist()
                results["normalized"]["test_out_large_n"] = test_out_large_n.squeeze().tolist()
            with open(os.path.join(params["log_dir"], "results.json"), "w") as f:
                json.dump(results, f)
        if not (str(seed) in results["unnormalized"]):
            params["model"]["normalized"] = False
            model_normalized, train_out, test_out, test_out_large_n = train(params)
            mse, rank_corr = eval(model_normalized, params, test_n_range)
            results["unnormalized"][str(seed)] = {"mse": mse, "rank_corr": rank_corr}
            if seed == 0:
                results["unnormalized"]["train_out"] = train_out.squeeze().tolist()
                results["unnormalized"]["test_out"] = test_out.squeeze().tolist()
                results["unnormalized"]["test_out_large_n"] = test_out_large_n.squeeze().tolist()
            with open(os.path.join(params["log_dir"], "results.json"), "w") as f:
                json.dump(results, f)

    plot_size_generalization(results, params)
    plot_output(results, params)
