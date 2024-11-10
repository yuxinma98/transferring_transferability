import torch
import os
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import Planetoid

class SubsampledDataset(InMemoryDataset):
    def __init__(self, root, dataset_name, n_samples, n_nodes, transform=None, pre_transform=None):
        os.makedirs(root, exist_ok=True)
        self.dataset_name = dataset_name
        self.n_samples = n_samples
        self.n_nodes = n_nodes
        super(SubsampledDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.dataset_name]

    @property
    def processed_file_names(self):
        return [f"{self.dataset_name}_{self.n_samples}_{self.n_nodes}.pt"]

    def process(self):
        if self.dataset_name == "Cora":
            data = Planetoid(self.root, "Cora")[0]
        else:
            raise ValueError(f"Dataset {self.dataset_name} is not supported.")

        W = to_dense_adj(data.edge_index)[:, data.train_mask, :][:, :, data.train_mask]
        f = data.x[data.train_mask, :]
        target = data.y[data.train_mask]
        n = W.shape[-1]
        z = torch.randint(0, n, (self.n_samples, self.n_nodes))
        data_list = []
        for i in range(self.n_samples):
            A = torch.zeros(self.n_nodes, self.n_nodes)
            X = torch.zeros(self.n_nodes, f.shape[-1])
            for j in range(self.n_nodes):
                X[j, :] = f[z[i, j], :]
                for k in range(self.n_nodes):
                    A[j, k] = W[:, z[i, j], z[i, k]]
            data = Data(x=X, edge_index=dense_to_sparse(A)[0], y=target[z[i]])
            data.A = A.unsqueeze(dim=0)
            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])
