import os
import argparse
import pytorch_lightning as pl

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
    test_mse = []
    for N in np.arange(1000,5000,500):
        dataset = PopStatsDataset(fname = os.path.join(params["data_dir"], f'task{params["task_id"]}/test_{N}.mat'))
        y_pred = model.predict(dataset.X)
        test_mse.append(mse(y_pred, dataset.y))
    return test_mse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_size", type=int, default=500)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=50)
    parser.add_argument("--set_channels", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=1000)

    args = parser.parse_args()

    params = {
        #logger parameters
        "project": "anydim_transferability",
        "name": "DeepSet",
        "logger": True,
        "log_checkpoint": False,
        "log_model": None,
        "log_dir": os.path.join(CURRENT_DIR, "log/"),
        # data parameters
        "data_dir": os.path.join(CURRENT_DIR, "generator/data"),
        "training_size": args.training_size,
        "batch_size": 128,
        # model parameters
        "model":{
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
        "training_seed":42,
    }
    if not os.path.exists(params["log_dir"]):
        os.makedirs(params["log_dir"])
    if not os.path.exists(params["data_dir"]):
        os.makedirs(params["data_dir"])
    
    for task_id in [1,2,3,4]:
        params["task_id"] = task_id
        
        mse_normalized_list = []
        mse_unnormalized_list = []
        for trial in range(args.num_trials):
            params["model"]["normalized"] = True
            params["training_seed"] = trial + 1
            model_normalized = train(params)
            mse_normalized = eval(model_normalized, params)
            mse_normalized_list.append(mse_normalized[-1])

            params["model"]["normalized"] = False
            model_unnormalized = train(params)
            mse_unnormalized = eval(model_unnormalized, params)
            mse_unnormalized_list.append(mse_unnormalized[-1])

        mean_mse_normalized = np.mean(mse_normalized_list, axis=0)
        mean_mse_unnormalized = np.mean(mse_unnormalized_list, axis=0)
        min_mse_normalized = np.min(mse_normalized_list, axis=0)
        max_mse_normalized = np.max(mse_normalized_list, axis=0)
        min_mse_unnormalized = np.min(mse_unnormalized_list, axis=0)
        max_mse_unnormalized = np.max(mse_unnormalized_list, axis=0)
        
        plt.plot(np.arange(1000,5000,500), mean_mse_normalized, label='Normalized')
        plt.fill_between(np.arange(1000,5000,500), min_mse_normalized, max_mse_normalized, alpha=0.3)
        plt.plot(np.arange(1000,5000,500), mean_mse_unnormalized, label='Unnormalized')
        plt.fill_between(np.arange(1000,5000,500), min_mse_unnormalized, max_mse_unnormalized, alpha=0.3)
        plt.xlabel('Test set size (N)')
        plt.ylabel('Test MSE')
        plt.title(f'Task {params["task_id"]}')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(os.path.join(params["log_dir"], f'size_generalization/task{params["task_id"]}_plot.png'))
