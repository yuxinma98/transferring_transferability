import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import torch_geometric.utils as pyg_utils

from data import HomDensityDataset
from model import GNN


def train(params):
    pl.seed_everything(params["training_seed"])
    model = GNNTrainingModule(params)
    model_checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}-{val_loss:.2f}",
        save_last=True,
        mode="min",
        monitor="val_loss",
    )
    if params["logger"]:
        logger = WandbLogger(
            project=params["project"],
            name=params["name"],
            log_model=params["log_checkpoint"],
            save_dir=params["log_dir"],
        )
        logger.watch(model, log=params["log_model"], log_freq=50)
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
    return model


class GNNTrainingModule(pl.LightningModule):
    def __init__(self, params):
        super(GNNTrainingModule, self).__init__()
        self.save_hyperparameters(params)
        self.params = params
        self.loss = nn.MSELoss()

    def prepare_data(self):
        self.dataset = HomDensityDataset(
            root=self.params["data_dir"],
            N=self.params["n_graphs"],
            n=self.params["n_nodes"],
            **self.params
        )
        # self.params["model"]["in_channels"] = (
        #     self.params["feature_dim"] + 1
        #     if self.params["task"] == "conditional_triangle"
        #     else self.params["feature_dim"]
        # )
        self.params["model"]["in_channels"] = 1
        self.params["model"]["out_channels"] = 1
        self.model = GNN(**self.params["model"])

    def setup(self, stage=None):
        if stage == "fit" or stage == None:
            train_fraction = 1 - self.params["val_fraction"] - self.params["test_fraction"]

            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset,
                [train_fraction, self.params["val_fraction"], self.params["test_fraction"]],
                generator=torch.Generator().manual_seed(self.params["data_seed"]),
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.params["batch_size"], shuffle=True, drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.params["batch_size"])

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.params["batch_size"])

    def forward(self, data):
        A = pyg_utils.to_dense_adj(data.edge_index, batch=data.batch)
        x, mask = pyg_utils.to_dense_batch(data.x, batch=data.batch)
        return self.model(A, x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.params["lr"], weight_decay=self.params["weight_decay"]
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = self.loss(out.reshape(-1), batch.y)
        self.log("train_loss", loss, batch_size=self.params["batch_size"])
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = self.loss(out.reshape(-1), batch.y)
        self.log("val_loss", loss, batch_size=self.params["batch_size"])
        return loss

    def test_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = self.loss(out.reshape(-1), batch.y)
        self.log("test_loss", loss, batch_size=self.params["batch_size"])
        return loss
