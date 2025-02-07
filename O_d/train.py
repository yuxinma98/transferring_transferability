import wandb
import os
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib
import matplotlib.pyplot as plt
from torchmetrics.regression import SpearmanCorrCoef

from Anydim_transferability.O_d.data import GWLBDataModule
from Anydim_transferability.O_d.model import SiameseRegressor


def train(params, record_output=False):
    pl.seed_everything(params["training_seed"])
    model = OdSnTrainingModule(params)
    data = GWLBDataModule(
        fname=f"{params['data_dir']}/GWLB_points{params['point_cloud_size']}_classes[2, 7].pkl",
        model_name=params["model"]["model_name"],
    )
    model_checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}-{val_loss:.2f}",
        save_last=True,
        mode="min",
        monitor="train_loss",
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
        devices=1,
        callbacks=[model_checkpoint],
        max_epochs=params["max_epochs"],
        logger=logger if params["logger"] else None,
        enable_progress_bar=True,
    )
    trainer.fit(model, datamodule=data)
    if params["logger"]:
        logger.experiment.unwatch(model)
    trainer.test(model, datamodule=data, verbose=True, ckpt_path="best")
    best_train_loss = model_checkpoint.best_model_score.item()
    wandb.finish()

    if record_output:  # only record outputs for the first trial
        large_n_data = GWLBDataModule(
            fname=f"{params['data_dir']}/GWLB_points{params['large_point_cloud_size']}_classes[2, 7].pkl",
            model_name=params["model"]["model_name"],
        )
        large_n_data.prepare_data()
        model.eval()
        with torch.no_grad():
            train_pred = model.predict(data.X_train)
            test_pred = model.predict(data.X_test)
            test_pred_large_n = model.predict(large_n_data.X_test)
        return model, best_train_loss, train_pred, test_pred, test_pred_large_n
    else:
        return model, best_train_loss, None, None, None


class OdSnTrainingModule(pl.LightningModule):
    def __init__(self, params):
        super(OdSnTrainingModule, self).__init__()
        self.save_hyperparameters(params)
        self.params = params
        self.loss = nn.MSELoss()
        self.spearman = SpearmanCorrCoef().to(self.device)
        self.model = SiameseRegressor(params["model"])

    def forward(self, data):
        return self.model(data[0].to(self.device), data[1].to(self.device))

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.params["lr"], weight_decay=self.params["weight_decay"]
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        dist_true_train = self.trainer.datamodule.dist_true_train.to(self.device)
        loss = self.loss(out, dist_true_train)
        rank_corr = self.spearman(
            out.reshape(-1, 1),
            dist_true_train.reshape(-1, 1),
        )
        self.log("train_loss", loss)
        self.log("train_rank_corr", rank_corr)
        return loss

    def validation_step(self, batch, batch_idx):
        pass
        # out = self.forward(batch)
        # loss = self.loss(out, self.dist_true_train.to(out.device))
        # self.log("val_loss", loss)
        # return loss

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.forward(data)
        return y_pred

    def on_validation_end(self):
        super().on_validation_end()
        X_train = self.trainer.datamodule.X_train
        dist_true_train = self.trainer.datamodule.dist_true_train.to(self.device)
        if self.current_epoch % 50 == 0:
            y_pred = self.predict(X_train).cpu()
            self._plot_correlation(
                dist_true_train.cpu().numpy(),
                y_pred.flatten(),
                figname="current_status.png",
            )

    def test_step(self, batch, batch_idx):
        dist_true_test = self.trainer.datamodule.dist_true_test.to(self.device)
        out = self.forward(batch)
        loss = self.loss(out, dist_true_test)
        rank_corr = self.spearman(out.reshape(-1, 1), dist_true_test.reshape(-1, 1))
        self.log("test_loss", loss)
        self.log("test_rank_corr", rank_corr)
        return loss

    def on_test_end(self):
        super().on_validation_end()
        X_test = self.trainer.datamodule.X_test
        dist_true_test = self.trainer.datamodule.dist_true_test.to(self.device)
        y_pred = self.predict(X_test).cpu()
        self._plot_correlation(
            dist_true_test.squeeze().cpu(),
            y_pred.flatten(),
            figname="test_status.png",
        )
        self._plot_prediction(
            dist_true_test.squeeze().cpu(),
            y_pred.flatten(),
            figname="test_pred.png",
        )
        if self.params.get("logger", True):
            self.logger.log_image(
                key="test_plot",
                images=[os.path.join(self.params["log_dir"], "test_status.png")],
            )
            self.logger.log_image(
                key="test_pred", images=[os.path.join(self.params["log_dir"], "test_pred.png")]
            )

    def _plot_correlation(
        self,
        y_truth,
        y_pred,
        figname="train_status.png",
    ):
        font = {"size": 14}
        matplotlib.rc("font", **font)
        scale = 0.5
        plt.figure(figsize=(10 * scale, 7.5 * scale))
        plt.scatter(y_truth, y_pred, label="train", alpha=0.75)
        plt.plot(
            np.linspace(0, 1.8, 100), np.linspace(0, 1.8, 100), ls=":", color="gray", alpha=0.5
        )
        plt.xlabel(r"Target: ", fontsize=15)
        plt.ylabel(r"Prediction", fontsize=15)
        plt.title("Training Set")
        plt.legend(fontsize=12)
        plt.xlim(0, 1.8)
        plt.ylim(0, 1.8)
        plt.gca().set_aspect("equal")
        plt.tight_layout()
        plt.savefig(os.path.join(self.params["log_dir"], figname))
        plt.close()

    def _plot_prediction(self, y_truth, y_pred, figname="test_pred.png"):
        fig, axs = plt.subplots(ncols=2)
        axs[0].imshow(
            torch.unflatten(y_truth, 0, (40, 40)), cmap="binary", origin="lower", vmin=0, vmax=2
        )
        axs[1].imshow(
            torch.unflatten(y_pred, 0, (40, 40)), cmap="binary", origin="lower", vmin=0, vmax=2
        )
        axs[0].set_title(r"Target: ")
        axs[1].set_title(r"Pred: ")
        for ax in axs:
            ax.set_xlabel("Pointclouds (class 2)")
            ax.set_ylabel("Pointclouds (class 7)")
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        fig.suptitle("Test Set", y=0.75)
        plt.savefig(os.path.join(self.params["log_dir"], figname))
        plt.close()
