import wandb
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch_geometric as pyg
import torchmetrics
from torch_geometric.datasets import planetoid
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model import GNN


def train(params):
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


class GNNTrainingModule(pl.LightningModule):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.save_hyperparameters(params) # log hyperparameters in wandb
        self.model = GNN(**params["model"])
        self.params = params

        self.task = params["model"]["task"]
        if self.task == "classification":
            self.loss = nn.CrossEntropyLoss()
            self.metric = torchmetrics.Accuracy(task="multiclass", num_classes=params["model"]["out_channels"])
            self.metric_name = "acc"
        elif self.task == "regression":
            self.loss = nn.MSELoss()
            self.metric = torchmetrics.MeanSquaredError
            self.metric_name = "mse"

    def prepare_data(self):
        dataset = planetoid.Planetoid(root="data/", name="Cora", split="public")
        data = dataset[0]
        data.A = pyg.utils.to_dense_adj(data.edge_index).squeeze(dim=0)
        self.data = data

    def train_dataloader(self):
        return pyg.loader.DataLoader([self.data], batch_size=1)

    def val_dataloader(self):
        return pyg.loader.DataLoader([self.data], batch_size=1)

    def test_dataloader(self):
        return pyg.loader.DataLoader([self.data], batch_size=1)

    def forward(self, data: pyg.data.Data, mode: str) -> torch.Tensor:
        if mode == "train":
            mask = data.train_mask
        elif mode == "val":
            mask = data.val_mask
        elif mode == "test":
            mask = data.test_mask

        A = data.A[mask, mask]  # n x n
        X = data.x[mask, mask]  # n x D1
        out = self.model(A.unsqueeze(0), X.unsqueeze(0)).squeeze(0)

        # Compute predictions
        if self.task == "classification":
            pred = torch.argmax(out, dim=-1)
        elif self.task == "regression":
            pred = out
        return out, pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.params["lr"],
            betas=(0.9, 0.999),
            weight_decay=self.params["weight_decay"],
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self.params["lr_patience"]
        )
        scheduler = {
            "scheduler": sch,
            "monitor": "val_loss",
            "frequency": 1,
            "interval": "epoch",
        }
        return [optimizer], [scheduler]

    def training_step(self, batch: pyg.data.Data, batch_idx) -> torch.Tensor:
        loss, metric = self._compute_loss_and_metrics(batch, mode="train")
        self.log_dict(
            {"train_loss": loss, f"train_{self.metric_name}": metric}, batch_size=len(batch)
        )
        return loss

    def validation_step(self, batch: pyg.data.Data, batch_idx) -> None:
        loss, metric = self._compute_loss_and_metrics(batch, mode="val")
        self.log_dict({"val_loss": loss, f"val_{self.metric_name}": metric}, batch_size=len(batch))

    def on_test_start(self):
        super().on_test_start()
        self.test_metric = {}

    def test_step(self, batch: pyg.data.Data, batch_idx) -> None:
        loss, metric = self._compute_loss_and_metrics(batch, mode="test")
        self.test_metric = metric
        self.log_dict(
            {f"test_loss": loss, f"test_{self.metric_namex}": metric}, batch_size=len(batch)
        )

    def _compute_loss_and_metrics(self, data: pyg.data.Data, mode: str="train"):
        try:
            mask = getattr(data, f"{mode}_mask")
        except AttributeError:
            raise f"Unknown forward mode: {mode}"

        out, pred = self.forward(data, mode)
        loss = self.loss(out[mask], data.y[mask])
        metric = self.metric(pred[mask], data.y[mask])
        return loss, metric
