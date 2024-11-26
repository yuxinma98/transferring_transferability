import torch
import os
from tqdm import tqdm
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch.utils.data import Dataset


class SubsampledDataset(Dataset):
    def __init__(self, root, model, dataset_name, n_samples, n_nodes):
        self.n_samples = n_samples
        self.n_nodes = n_nodes
        if dataset_name == "Cora":
            data = Planetoid(root, "Cora")[0]
        elif dataset_name == "PubMed":
            data = Planetoid(root, "PubMed")[0]
        else:
            raise ValueError(f"Dataset {self.dataset_name} is not supported.")
        data.A = to_dense_adj(data.edge_index)
        self.W = data.A
        self.f = data.x
        with torch.no_grad():
            self.target = model(data).detach()
        self.n = self.W.shape[-1]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        z = torch.randint(0, self.n, (self.n_nodes,))
        data = Data(
            x=self.f[z, :],
            edge_index=dense_to_sparse(self.W[:, z, :][:, :, z])[0],
            A=self.W[:, z, :][:, :, z],
            y=self.target[z, :],
        )
        return data
