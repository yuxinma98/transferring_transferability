import torch
import random
import os
from typing import Union
import pytorch_lightning as pl
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.utils as pyg_utils
from torch_geometric.datasets import SNAPDataset
from torch.utils.data import Dataset


class SBM_GaussianDataset(Dataset):

    def __init__(
        self, root: Union[str, os.PathLike], N: int = int(1e3), d: int = 2, K: int = 3
    ) -> None:
        """Synthetic dataset from Stochastic Block Model (SBM) and Gaussian features.
        Features are iid Gaussian within the same cluster, with different means and covariances for different clusters.

        Args:
            root (Union[str, os.PathLike]): data directory
            N (int, optional): Number of nodes. Defaults to int(1e3).
            d (int, optional): Number of features. Defaults to 2.
            K (int, optional): Number of clusters in SBM. Defaults to 3.
        """
        super().__init__()
        self.root = root
        self.N = N
        self.d = d
        self.K = K
        self.prepare_data()

    def prepare_data(self):
        fname = os.path.join(self.root, f"Synthetic/SBM_Gaussian_{self.N}_{self.d}_{self.K}.pt")
        if os.path.exists(fname):
            self.data = torch.load(fname)
        else:
            os.makedirs(os.path.dirname(fname), exist_ok=True)

            # Define SBM and Gaussian parameters
            uniform_sampler = torch.distributions.Uniform(0.1, 0.9)
            # randomly generate a symmetric K x K probability matrix for SBM, indicating the probability of edge between clusters
            ps = uniform_sampler.sample((self.K, self.K))
            ps = (ps + ps.transpose(-1, -2)) / 2
            # randomly generate mean and covariance matrix for Gaussian
            mu = torch.randn(self.K, self.d)  # K x d
            L = torch.randn(self.K, self.d, self.d)
            cov = torch.matmul(
                L, L.transpose(-1, -2)
            )  # K x d x d, covariance matrix should be positive semi-definite

            # Generate graph
            z = torch.randint(0, self.K, (self.N,))  # randomly assign cluster for each node
            z_one_hot = torch.nn.functional.one_hot(z, num_classes=self.K).float()  # N x K
            prob_matrix = z_one_hot @ ps
            prob_matrix = prob_matrix @ z_one_hot.transpose(-1, -2)  # N x N
            A = torch.distributions.Bernoulli(prob_matrix).sample()
            A = A.tril(diagonal=0) + A.tril(diagonal=-1).transpose(-1, -2)  # make A symmetric

            # Generate features
            X_distributions = torch.distributions.MultivariateNormal(mu[z], cov[z])
            X = X_distributions.sample()

            # Generate normalized degrees as target
            y = A @ torch.ones(self.N, 1) / self.N
            y = y.squeeze()

            # Split data into train, validation, and test set
            train_mask = torch.zeros(self.N, dtype=torch.bool)
            val_mask = torch.zeros(self.N, dtype=torch.bool)
            test_mask = torch.zeros(self.N, dtype=torch.bool)
            train_mask[: int(0.8 * self.N)] = True
            val_mask[int(0.8 * self.N) : int(0.9 * self.N)] = True
            test_mask[int(0.9 * self.N) :] = True

            self.data = Data(
                x=X.unsqueeze(0),
                edge_index=pyg_utils.dense_to_sparse(A)[0],
                A=A.unsqueeze(0),
                y=y,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
            )
            torch.save(self.data, fname)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data


class HomDensityDataset(InMemoryDataset):

    def __init__(self, root, N, n, graph_model, task, **kwargs):
        self.N = N
        self.n = n
        self.graph_model = graph_model
        self.task = task
        if self.graph_model == "ER":
            self.p = [0.3, 0.7]
        elif self.graph_model == "SBM":
            self.ps = torch.tensor([[0.8, 0.15, 0.05], [0.15, 0.7, 0.25], [0.05, 0.25, 0.9]])
        elif self.graph_model == "Sociality":
            self.kernel = lambda x, y: 1 / (1 + torch.exp(-x * y))
        elif self.graph_model == "spiked":
            self.kernel = lambda x, y: torch.exp(-0.3 * (x - y) ** 2)
        super(HomDensityDataset, self).__init__(root=root, transform=None, pre_transform=None)
        self.data, self.slices = self.process()

    @property
    def processed_file_names(self):
        return [f"{self.graph_model}_{self.task}_{self.N}_{self.n}.pt"]

    def process(self):
        data_list = []

        for i in range(self.N):
            if isinstance(self.n, int):
                n = self.n
            else:
                n = random.randint(self.n[0], self.n[1])

            if self.graph_model == "ER":
                if isinstance(self.p, float):
                    p = self.p
                else:
                    p = random.uniform(self.p[0], self.p[1])
                edge_index = pyg_utils.erdos_renyi_graph(n, p)
                A = pyg_utils.to_dense_adj(edge_index).squeeze()
            elif self.graph_model == "SBM":
                z = torch.randint(0, 3, (n,))
                z_one_hot = torch.nn.functional.one_hot(z, num_classes=3).float()
                prob_matrix = z_one_hot @ self.ps
                prob_matrix = prob_matrix @ z_one_hot.transpose(-1, -2)
                A = torch.distributions.Bernoulli(prob_matrix).sample()
                A = A.tril(diagonal=-1) + A.tril(diagonal=-1).transpose(-1, -2)
                edge_index = pyg_utils.dense_to_sparse(A.unsqueeze(0))[0]
            elif self.graph_model == "Sociality" or self.graph_model == "spiked":
                uniform_sampler = torch.distributions.Uniform(0, 1)
                z = uniform_sampler.sample((n,))
                prob_matrix = self.kernel(z.unsqueeze(1), z.unsqueeze(0))
                A = torch.distributions.Bernoulli(prob_matrix).sample()
                A = A.tril(diagonal=0) + A.tril(diagonal=-1).transpose(-1, -2)
                edge_index = pyg_utils.dense_to_sparse(A.unsqueeze(0))[0]

            # x = torch.randn((n, self.d))  # Random node features, n x d

            if self.task == "degree":
                y = pyg_utils.degree(edge_index[0]) / n
            elif self.task == "triangle" or self.task == "conditional_triangle":
                mask = z < 0.3
                A[mask, :][:, mask] = 0
                y = (A @ A @ A).diag() / (n**2)
            elif self.task == "4-cycle":
                y = (A @ A @ A @ A).diag() / (n**3)

            data = Data(x=z.unsqueeze(-1), edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices


class SubsampledDataset(Dataset):

    def __init__(
        self,
        model: Union[torch.nn.Module, pl.LightningModule],
        n_samples: int,
        n_nodes: int,
    ) -> None:
        """Subsampled dataset for transferability experiment."""
        self.n_samples = n_samples
        self.n_nodes = n_nodes
        data = model.dataset[0]
        with torch.no_grad():
            self.target = model(data).detach()
        self.W = pyg_utils.to_dense_adj(data.edge_index)
        self.f = data.x
        self.N = self.W.shape[-1]

        self.generate_dataset()

    def subsample_data(self):
        z = torch.randint(0, self.N, (self.n_nodes,))
        return Data(
            x=self.f[z, :],
            edge_index=pyg_utils.dense_to_sparse(self.W[:, z, :][:, :, z])[0],
        )

    def generate_dataset(self):
        self.dataset = []
        for _ in range(self.n_samples):
            sample = self.subsample_data()
            self.dataset.append(sample)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.dataset[idx]
