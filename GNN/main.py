import wandb
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from train import GNNTrainingModule

def train(params):
    # fix seed
    pl.seed_everything(params["training_seed"])
    model = GNNTrainingModule(params)
    model_checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}-{val_loss:.2f}",
        save_last=True,
        mode="max",
        monitor="val_acc",
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
    trainer.fit(model)
    if params["logger"]:
        logger.experiment.unwatch(model)
    trainer.test(model, verbose=True, ckpt_path="best")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_fraction", type=float, default=0.1)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--hidden_channels", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--max_epochs", type=int, default=300)

    args = parser.parse_args()
    params = {
        #logger parameters
        "project": "anydim_transferability",
        "name": "GNN",
        "logger": True,
        "log_checkpoint": False,
        "log_model": None,
        "log_dir": "log/",
        # data parameters
        "sample_fraction": args.sample_fraction,
        "data_seed": 1,
        # model parameters
        "model":{
            "in_channels":1433,
            "hidden_channels": args.hidden_channels,
            "num_layers": args.num_layers,
            "out_channels": 7, # classification into 7 classes
            "task": "classification",
        },
        # training parameters
        "lr": args.lr,
        "lr_patience": 10,
        "weight_decay": 0.1,
        "max_epochs": args.max_epochs,
        "training_seed":42,
    }
    for i in range(args.num_trials):
        params["training_seed"] = i
        train(params)