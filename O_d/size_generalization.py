import os
import argparse
from torchmetrics import MeanSquaredError, SpearmanCorrCoef
import torch
import json
import numpy as np
import matplotlib.pyplot as plt

from Anydim_transferability.O_d.train import train, GWLBDataModule
from Anydim_transferability.O_d import color_dict, data_dir, plot_model_names
from Anydim_transferability import typesetting, plot_dir

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
typesetting()


def eval(model, params):
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
            model_name=params["model"]["model_name"],
        )
        data.prepare_data()
        with torch.no_grad():
            y_pred = model.predict(data.X_test)
        test_mse.append(float(mse(y_pred, data.dist_true_test.to(model.device))))
        test_rank_corr.append(
            float(
                spearman(y_pred.reshape(-1, 1), data.dist_true_test.to(model.device).reshape(-1, 1))
            )
        )
    return test_mse, test_rank_corr


def train_and_eval(params, args, model_name):
    results.setdefault(model_name, {})
    seed = 0
    for trial in range(args.num_trials):  # run multiple trials
        if str(trial) in results[model_name]:
            continue
        record_output = trial == 0
        best_model = None
        for i in range(5):  # for each trial, take the best out of 5 random runs
            seed = trial * 5 + i
            params["training_seed"] = seed
            model, train_loss, train_out, test_out, test_out_large_n = train(
                params, record_output=record_output
            )
            if best_model is None or train_loss < best_loss:
                best_model, best_loss, best_train_out, best_test_out, best_test_out_large_n = (
                    model,
                    train_loss,
                    train_out,
                    test_out,
                    test_out_large_n,
                )
        mse, rank_corr = eval(best_model, params)
        results[model_name][str(trial)] = {"mse": mse, "rank_corr": rank_corr}
        if trial == 0:
            results[model_name]["train_out"] = best_train_out.squeeze().tolist()
            results[model_name]["test_out"] = best_test_out.squeeze().tolist()
            results[model_name]["test_out_large_n"] = best_test_out_large_n.squeeze().tolist()
        with open(os.path.join(params["log_dir"], "results.json"), "w") as f:
            json.dump(results, f)


def plot_size_generalization(results, params):
    axes_rect = [0.19, 0.16, 0.8, 0.81]
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_axes(axes_rect)
    # First subplot for Test MSE
    for model_name in model_params.keys():
        mse = [results[model_name][str(trial)]["mse"] for trial in range(args.num_trials)]
        ax.plot(
            test_n_range,
            np.mean(mse, axis=0),
            "o-",
            label=plot_model_names[model_name],
            color=color_dict[model_name],
        )
        ax.fill_between(
            test_n_range,
            np.min(mse, axis=0),
            np.max(mse, axis=0),
            alpha=0.3,
            color=color_dict[model_name],
        )
    ax.set_yscale("log")
    ax.set_xlabel("Test pointcloud sizes (n)", fontsize=20)
    ax.set_ylabel("Test MSE", fontsize=20)
    ax.set_xticks(test_n_range)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)
    plt.legend(fontsize=13, loc="upper right")
    plt.savefig(os.path.join(plot_dir, "SVD-DS_plot.png"))
    plt.savefig(os.path.join(plot_dir, "SVD-DS_plot.pdf"))
    plt.close()


def plot_output(results, params):
    data = GWLBDataModule(
        fname=f"{params['data_dir']}/GWLB_points{params['point_cloud_size']}_classes[2, 7].pkl",
        model_name="SVD-DS",
    )
    data.prepare_data()

    plt.figure(figsize=(10, 5))

    # # First subplot for Training Set
    # # Define the proportion of data to sample
    sample_proportion = 0.3

    # plt.subplot(1, 3, 1)
    # for model_name in model_params.keys():
    #     # Randomly subsample indices
    #     num_samples = int(len(data.dist_true_train) * sample_proportion)
    #     sampled_indices = np.random.choice(len(data.dist_true_train), num_samples, replace=False)

    #     # Subsample the data
    #     sampled_true_train = data.dist_true_train[sampled_indices]
    #     sampled_train_out = np.array(results[model_name]["train_out"])[sampled_indices]

    #     plt.scatter(
    #         sampled_true_train,
    #         sampled_train_out,
    #         label=plot_model_names[model_name],
    #         alpha=0.4,
    #         s=20,
    #         marker="o",
    #         color=color_dict[model_name],
    #     )
    # plt.plot(np.linspace(0, 1.8, 100), np.linspace(0, 1.8, 100), ls=":", color="gray", alpha=0.5)
    # plt.xlabel("Target", fontsize=15)
    # plt.ylabel("Prediction", fontsize=15)
    # plt.title("Training Set \n (Pointcloud size $n=20$)", fontsize=18)
    # plt.legend(fontsize=12)
    # plt.xlim(0, 1.8)
    # plt.ylim(0, 1.8)
    # plt.gca().set_aspect("equal")

    # Second subplot for Test Set
    plt.subplot(1, 2, 1)
    for model_name in model_params.keys():
        # Randomly subsample indices
        num_samples = int(len(data.dist_true_test) * sample_proportion)
        sampled_indices = np.random.choice(len(data.dist_true_test), num_samples, replace=False)

        # Subsample the data
        sampled_true_test = data.dist_true_test[sampled_indices]
        sampled_test_out = np.array(results[model_name]["test_out"])[sampled_indices]

        plt.scatter(
            sampled_true_test,
            sampled_test_out,
            label=plot_model_names[model_name],
            alpha=0.4,
            s=20,
            marker="o",
            color=color_dict[model_name],
        )
    plt.plot(np.linspace(0, 1.8, 100), np.linspace(0, 1.8, 100), ls=":", color="gray", alpha=0.5)
    plt.xlabel("Target", fontsize=18)
    plt.ylabel("Prediction", fontsize=18)
    plt.title("Model output on \n pointcloud size $n_{test}=20$", fontsize=20)
    plt.legend(fontsize=13)
    plt.xlim(0, 1.8)
    plt.ylim(0, 1.8)
    plt.gca().set_aspect("equal")

    data = GWLBDataModule(
        fname=f"{params['data_dir']}/GWLB_points{params['large_point_cloud_size']}_classes[2, 7].pkl",
        model_name="SVD-DS",
    )
    data.prepare_data()
    # Third subplot for Test Set with large n
    plt.subplot(1, 2, 2)
    for model_name in model_params.keys():
        # Randomly subsample indices
        num_samples = int(len(data.dist_true_test) * sample_proportion)
        sampled_indices = np.random.choice(len(data.dist_true_test), num_samples, replace=False)

        # Subsample the data
        sampled_true_test = data.dist_true_test[sampled_indices]
        sampled_test_out = np.array(results[model_name]["test_out_large_n"])[sampled_indices]

        plt.scatter(
            sampled_true_test,
            sampled_test_out,
            label=plot_model_names[model_name],
            alpha=0.4,
            s=20,
            marker="o",
            color=color_dict[model_name],
        )
    plt.plot(np.linspace(0, 1.8, 100), np.linspace(0, 1.8, 100), ls=":", color="gray", alpha=0.5)
    plt.xlabel("Target", fontsize=18)
    plt.ylabel("Prediction", fontsize=18)
    plt.title("Model output on \n pointcloud size $n_{test}=500$", fontsize=20)
    plt.legend(fontsize=13)
    plt.xlim(0, 1.8)
    plt.ylim(0, 1.8)
    plt.gca().set_aspect("equal")
    plt.savefig(os.path.join(plot_dir, "SVD-DS_output_plot.png"))
    plt.savefig(os.path.join(plot_dir, "SVD-DS_output_plot.pdf"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment set-up
    parser.add_argument("--num_trials", type=int, default=10, help="Number of trials to run")
    # DeepSet model parameters
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--max_epochs", type=int, default=3000)

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
        "data_dir": data_dir,
        "point_cloud_size": 20,
        "large_point_cloud_size": 500,
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

    model_params = {
        "SVD-DS": {
            "hid_dim": 16,
            "out_dim": 16,
        },
        "SVD-DS (Normalized)": {
            "hid_dim": 16,
            "out_dim": 16,
        },
        "DS-CI (Normalized)": {"hid_dim": 10, "out_dim": 10},
        # "DS-CI (Compatible)": {"hid_dim": 10, "out_dim": 10},
        # "OI-DS (Normalized)": {"hid_dim": 12, "out_dim": 12},
    }
    test_n_range = [20, 100, 200, 300, 500]
    for model_name in model_params.keys():
        params["model"] = model_params[model_name]
        params["model"]["model_name"] = model_name
        train_and_eval(params, args, model_name)

    plot_size_generalization(results, params)
    plot_output(results, params)
