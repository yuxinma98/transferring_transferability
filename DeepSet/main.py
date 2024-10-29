import wandb
import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from train import DeepSetTrainingModule
from data import PopStatsDataModule

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
    
def train(params):
    pl.seed_everything(params["training_seed"])
    data = PopStatsDataModule(data_dir=params["data_dir"],
                              task_id = params["task_id"],
                              exp_id = params["exp_id"],
                              batch_size = params["batch_size"])
    data.setup()
    params["model"]["in_channels"] = data.d
    model = DeepSetTrainingModule(params)
    model_checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}-{val_loss:.2f}",
        save_last=True,
        mode="min",
        monitor="val_mse",
    )
    if params["logger"]:
        logger = WandbLogger(
            project=params["project"], name=params["name"], log_model=params["log_checkpoint"], save_dir=params["log_dir"]
        )
        logger.watch(model, log = params["log_model"], log_freq=50)
    trainer = pl.Trainer(
        callbacks=[model_checkpoint],
        devices=1,
        max_epochs=params["max_epochs"],
        logger=logger if params["logger"] else None,
        enable_progress_bar=True,
    )
    trainer.fit(model,datamodule=data)
    if params["logger"]:
        logger.experiment.unwatch(model)
    trainer.test(model, datamodule=data, verbose=True, ckpt_path="best")
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, choices=[1,2,3,4])
    parser.add_argument("--exp_id", type=int, choices=list(range(7,18)))
    parser.add_argument("--normalized", type=str2bool, default=True)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=50)
    parser.add_argument("--set_channels", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_trials", type=int, default=5)
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
        "task_id": args.task_id,
        "exp_id": args.exp_id,
        "batch_size": 128,
        # model parameters
        "model":{
            "normalized": args.normalized,
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
    for i in range(args.num_trials):
        params["training_seed"] = i
        train(params)