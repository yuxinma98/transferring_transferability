import torch
import os
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch.utils.data import Dataset


class SBM_GaussianDataset(Dataset):

    def __init__(self, root, N=int(1e3), d=2, K=3):
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
            uniform_sampler = torch.distributions.Uniform(0.1, 0.9)
            ps = uniform_sampler.sample((3, 3))
            L = torch.randn(3, self.d, self.d)
            mu = torch.randn(3, self.d)
            cov = torch.matmul(L, L.transpose(-1, -2))

            z = torch.randint(0, 3, (self.N,))
            z_one_hot = torch.nn.functional.one_hot(z, num_classes=3).float()
            prob_matrix = z_one_hot @ ps
            prob_matrix = prob_matrix @ z_one_hot.transpose(-1, -2)
            A = torch.distributions.Bernoulli(prob_matrix).sample()
            A = A.tril(diagonal=-1) + A.tril(diagonal=-1).transpose(-1, -2)
            X_distributions = torch.distributions.MultivariateNormal(mu[z], cov[z])
            X = X_distributions.sample()
            y = A @ torch.ones(self.N, 1) / self.N
            y = y.squeeze()

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
    def __init__(self, root, model, dataset_name, n_samples, n_nodes):
        self.n_samples = n_samples
        self.n_nodes = n_nodes
        if dataset_name == "Cora":
            data = Planetoid(root, "Cora")[0]
        elif dataset_name == "PubMed":
            data = Planetoid(root, "PubMed")[0]
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
        self.W = data.A[:, rank, :][:, :, rank]
        self.f = data.x[rank, :]
        self.target = target[rank, :]
        self.n = self.W.shape[-1]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        z = torch.randint(0, self.n, (self.n_nodes,))
        z = z[torch.argsort(z)]

        index = torch.arange(self.n)
        z_expanded = z.unsqueeze(0).expand(self.n, -1)  # n x n_nodes
        index_expanded = index.unsqueeze(1).expand(-1, self.n_nodes)  # n x n_nodes
        distances = torch.abs(index_expanded - z_expanded)
        closest_indices = torch.argmin(distances, dim=1).unsqueeze(0)  # 1 x n

        data = Data(
            x=self.f[z, :],
            edge_index=dense_to_sparse(self.W[:, z, :][:, :, z])[0],
            A=self.W[:, z, :][:, :, z],
            indices=closest_indices,
        )
        return data
