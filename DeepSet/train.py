import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib
import os
import wandb
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import DeepSet
from data import PopStatsDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

class DeepSetTrainingModule(pl.LightningModule):
    def __init__(self, params: dict) -> None:
        super().__init__()
        self.save_hyperparameters(params)
        self.params = params
        self.model = DeepSet(**params["model"])
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.params["lr"],
            weight_decay=self.params["weight_decay"],
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=self.params["lr_patience"]
        )
        scheduler = {
            "scheduler": sch,
            "monitor": "val_mse",
            "frequency": 1,
            "interval": "epoch",
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.model(X)
        loss = self.l2(y_pred, y)
        self.log('train_loss', loss)
        return loss

    def on_validation_start(self):
        super().on_validation_start()
        self.val_t = []
        self.val_y_preds = []

    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.model(X)
        mae = self.l1(y_pred, y)
        mse = self.l2(y_pred, y)
        self.log('val_mae', mae)
        self.log('val_mse', mse)

    def on_validation_end(self):
        super().on_validation_end()
        if self.current_epoch % 50 == 0:
            truth = self.trainer.datamodule.truth
            val = self.trainer.datamodule.val_dataset
            y_pred = self.predict(val.X)
            font = {"size": 14}
            matplotlib.rc("font", **font)
            scale = 0.5
            plt.figure(figsize=(10 * scale, 7.5 * scale))
            plt.plot(truth.t, truth.y)
            plt.plot(val.t.tolist(), y_pred.tolist(), "x")
            plt.xlabel("Index")
            plt.ylabel("Statistics")
            plt.legend(["Truth", "DeepSet"], loc=3, fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(self.params["log_dir"], "current_status.png"))
            plt.close()
            if self.params.get("logger", True):
                self.logger.log_image(
                    key="current_status",
                    images=[os.path.join(self.params["log_dir"], "current_status.png")],
                )

    def test_step(self, batch, batch_idx):
        X, y = batch
        y_pred = self.model(X)
        mae = self.l1(y_pred, y)
        mse = self.l2(y_pred, y)
        self.log('test_mae', mae)
        self.log('test_mse', mse)

    def predict(self, X):
        with torch.no_grad():
            return self.model(X.to(self.device))


def train(params, stopping_threshold=False):
    pl.seed_everything(params["training_seed"])
    data = PopStatsDataModule(data_dir=params["data_dir"],
                              task_id = params["task_id"],
                              batch_size = params["batch_size"],
                              training_size = params["training_size"])
    data.setup()
    params["model"]["in_channels"] = data.d
    model = DeepSetTrainingModule(params)
    model_checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}-{val_loss:.2f}",
        save_last=True,
        mode="min",
        monitor="val_mse",
    )
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_mse", patience=params["max_epochs"], stopping_threshold=1e-3
    )
    if params["logger"]:
        logger = WandbLogger(
            project=params["project"], name=params["name"], log_model=params["log_checkpoint"], save_dir=params["log_dir"]
        )
        logger.watch(model, log = params["log_model"], log_freq=50)
    trainer = pl.Trainer(
        callbacks=[model_checkpoint, early_stopping] if stopping_threshold else [model_checkpoint],
        devices=1,
        max_epochs=params["max_epochs"],
        logger=logger if params["logger"] else None,
        enable_progress_bar=True,
    )
    trainer.fit(model,datamodule=data)
    if params["logger"]:
        logger.experiment.unwatch(model)
    trainer.test(model, datamodule=data, verbose=True, ckpt_path="best")

    test_data = data.test_dataset
    y_pred = model.predict(test_data.X)
    y_pred_out = [test_data.t.tolist(), y_pred.tolist()]
    return model, y_pred_out
