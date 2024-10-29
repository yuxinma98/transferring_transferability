import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import DeepSet

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
        truth = self.trainer.datamodule.truth
        val = self.trainer.datamodule.val_dataset
        y_pred = self.predict(val.X)
        font = {'size': 14}
        matplotlib.rc('font', **font)
        scale = 0.5
        plt.figure(figsize=(10*scale, 7.5*scale))
        plt.plot(truth.t, truth.y)
        plt.plot(val.t.tolist(), y_pred.tolist(), 'x')
        plt.xlabel('Index')
        plt.ylabel('Statistics')
        plt.legend(['Truth', 'DeepSet'], loc=3, fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.params["log_dir"], "current_status.png"))
        plt.close()
        if self.params.get("logger", True):
            self.logger.log_image(key = "current_status", images = [os.path.join(self.params["log_dir"], "current_status.png")])
    
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
