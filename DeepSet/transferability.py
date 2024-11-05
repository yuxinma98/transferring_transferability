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
    N = params["testing_size"]
    dataset = PopStatsDataset(fname = os.path.join(params["data_dir"], f'task{params["task_id"]}/test_{N}.mat'))
    y_pred = model.predict(dataset.X)
    return mse(y_pred, dataset.y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--testing_size", type=int, default=5000)
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

        params["model"]["normalized"] = True
        mse_normalized = []
        for training_size in range(500, 3000, 500):
            params["training_size"] = training_size
            model_normalized = train(params)
            mse_normalized.append(eval(model_normalized, params))


        params["model"]["normalized"] = False
        mse_unnormalized = []
        for training_size in range(500, 3000, 500):
            params["training_size"] = training_size
            model_unnormalized = train(params)
            mse_unnormalized.append(eval(model_unnormalized, params))
        model_unnormalized = train(params)
        mse_unnormalized = eval(model_unnormalized, params)
    
        plt.plot(np.arange(500,3000,500), mse_normalized, label='Normalized')
        plt.plot(np.arange(500,3000,500), mse_unnormalized, label='Unnormalized')
        plt.xlabel('Train set size (N)')
        plt.ylabel(f'Test MSE on N = {params["testing_size"]}')
        plt.title(f'Task {params["task_id"]}')
        plt.legend()
        plt.savefig(os.path.join(params["log_dir"], f'transferability/task{params["task_id"]}_plot.png'))   
