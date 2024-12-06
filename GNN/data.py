import torch
import os
from typing import Union
import pytorch_lightning as pl
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, SNAPDataset
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
            A = A.tril(diagonal=-1) + A.tril(diagonal=-1).transpose(-1, -2)  # make A symmetric

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
                x=X,
                edge_index=dense_to_sparse(A)[0],
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


class SubsampledDataset(Dataset):

    def __init__(
        self,
        root: Union[str, os.PathLike],
        model: Union[torch.nn.Module, pl.LightningModule],
        dataset_name: str,
        n_samples: int,
        n_nodes: int,
    ) -> None:
        """Subsampled dataset for transferability experiment.

        Args:
            root (Union[str, os.PathLike]): data directory
            model (Union[torch.nn.Module, pl.LightningModule]): GNN model to generate graph signals
            dataset_name (str): name of the dataset
            n_samples (int): number of graphs
            n_nodes (int): number of nodes in each graph

        Raises:
            ValueError: Raise error if dataset_name is not supported
        """
        self.n_samples = n_samples
        self.n_nodes = n_nodes
        if dataset_name == "Cora":
            data = Planetoid(root, "Cora")[0]
        elif dataset_name == "PubMed":
            data = Planetoid(root, "PubMed")[0]
        elif dataset_name == "facebook":
            data = SNAPDataset(root, "ego-facebook")[0]
        elif dataset_name == "SBM_Gaussian":
            data = SBM_GaussianDataset(root)[0]
        else:
            raise ValueError(f"Dataset {self.dataset_name} is not supported.")

        if dataset_name != "SBM_Gaussian":
            data.A = to_dense_adj(data.edge_index)

        with torch.no_grad():
            target = model(data).detach()
        degree = data.A.sum(-1)
        rank = torch.argsort(degree).squeeze()
        # use the ordered adjacency matrix as graph signal as step graphon to sample from
        self.W = data.A[:, rank, :][:, :, rank]
        self.f = data.x[rank, :]
        self.target = target[rank, :]
        self.n = self.W.shape[-1]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Sample n_nodes nodes with replacement
        z = torch.randint(0, self.n, (self.n_nodes,))
        # Sort the sampled nodes
        z = z[torch.argsort(z)]

        # for each original node (1-n), find the closest node in the sampled nodes
        index = torch.arange(self.n)
        z_expanded = z.unsqueeze(0).expand(self.n, -1)  # n x n_nodes
        index_expanded = index.unsqueeze(1).expand(-1, self.n_nodes)  # n x n_nodes
        distances = torch.abs(index_expanded - z_expanded)
        closest_indices = torch.argmin(distances, dim=1).unsqueeze(0)  # 1 x n

        # Construct one subsampled graph data
        data = Data(
            x=self.f[z, :],
            edge_index=dense_to_sparse(self.W[:, z, :][:, :, z])[0],
            A=self.W[:, z, :][:, :, z],
            indices=closest_indices,  # record how to map back to the graphon
        )
        return data
