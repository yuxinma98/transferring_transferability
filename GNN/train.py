import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch_geometric as pyg
import torchmetrics
from torch_geometric.datasets import planetoid
from model import GNN

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
        dataset = planetoid.Planetoid(root="data/", name="Cora", split="public", transform=pyg.transforms.NormalizeFeatures())
        data = dataset[0]
        data.A = pyg.utils.to_dense_adj(data.edge_index).squeeze(dim=0)
        self.data = data

        num_nodes = data.num_nodes
        sample_size = int(num_nodes * self.params["sample_fraction"])  # 1/10 of the nodes
        sampled_nodes = torch.randperm(num_nodes)[:sample_size]
        subgraph_edge_index, subgraph_edge_attr = pyg.utils.subgraph(sampled_nodes, data.edge_index, data.edge_attr)
        subgraph_data = data.clone()
        subgraph_data.edge_index = subgraph_edge_index
        subgraph_data.edge_attr = subgraph_edge_attr
        subgraph_data.x = data.x[sampled_nodes]
        subgraph_data.y = data.y[sampled_nodes]
        subgraph_data.train_mask = data.train_mask[sampled_nodes]
        subgraph_data.val_mask = data.val_mask[sampled_nodes]
        subgraph_data.test_mask = data.test_mask[sampled_nodes]
        subgraph_data.A = data.A[sampled_nodes, :][:, sampled_nodes]
        self.subgraph_data = subgraph_data
    
    def train_dataloader(self):
        return pyg.loader.DataLoader([self.subgraph_data], batch_size=1)
    
    def val_dataloader(self):
        return pyg.loader.DataLoader([self.subgraph_data], batch_size=1)
    
    def test_dataloader(self):
        return [pyg.loader.DataLoader([self.subgraph_data], batch_size=1),
                pyg.loader.DataLoader([self.data], batch_size=1)]
   
    def forward(self, data: pyg.data.Data) -> torch.Tensor:
        A = data.A # n x n
        X = data.x # n x D1
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
        self.log_dict({"train_loss": loss, 
                       f"train_{self.metric_name}": metric},
                       batch_size=len(batch))
        return loss

    def validation_step(self, batch: pyg.data.Data, batch_idx) -> None:
        loss, metric = self._compute_loss_and_metrics(batch, mode="val")
        self.log_dict({"val_loss": loss, 
                       f"val_{self.metric_name}": metric},
                       batch_size=len(batch))

    def on_test_start(self):
        super().on_test_start()
        self.test_metric = {}
    
    def test_step(self, batch: pyg.data.Data, batch_idx, dataloader_idx=0) -> None:
        loss, metric = self._compute_loss_and_metrics(batch, mode="test")
        self.test_metric[dataloader_idx] = metric
        dataset_names = {
            0: "subgraph",
            1: "full"
        }
        self.log_dict({f"test_loss_{dataset_names[dataloader_idx]}": loss,
                       f"test_{self.metric_name}_{dataset_names[dataloader_idx]}": metric},
                        batch_size=len(batch)) 

    def on_test_end(self):
        super().on_test_end()
        transferability = (self.test_metric[1] - self.test_metric[0])/self.test_metric[0]
        self.logger.experiment.log_metrics("transferability", transferability)

    def _compute_loss_and_metrics(self, data: pyg.data.Data, mode: str="train"):
        try:
            mask = getattr(data, f"{mode}_mask")
        except AttributeError:
            raise f"Unknown forward mode: {mode}"
        
        out, pred = self.forward(data)
        loss = self.loss(out[mask], data.y[mask])
        metric = self.metric(pred[mask], data.y[mask])
        return loss, metric