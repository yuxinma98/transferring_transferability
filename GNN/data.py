import torch
import random
import os
from typing import Union
import pytorch_lightning as pl
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.utils as pyg_utils
from torch.utils.data import Dataset


class HomDensityDataset(InMemoryDataset):

    def __init__(
        self,
        root: Union[str, os.PathLike],
        N: int,
        n: Union[int, tuple],
        graph_model: str,
        task: str,
        **kwargs,
    ):
        """
        Args:
            root (Union[str, os.PathLike]): directory to save the dataset
            N (int): Number of graphs
            n (Union[int, tuple]): Number of nodes
            graph_model (str): choice of graph generative model for the generation of data
            task (str): choice of task for learning
        """
        self.N = N
        self.n = n
        self.graph_model = graph_model
        self.task = task
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

            if self.graph_model == "SBM_Gaussian":
                # Randomly generate SBM parameters
                # K = random.randint(2, 10)  # Number of clusters
                K = 3
                p0 = random.random()
                p1 = random.random()
                ps = torch.tensor(
                    [[p0, p1, p1], [p1, p0, p1], [p1, p1, p0]]
                )  # 3 x 3 probability matrix for SBM

                # Generate graph
                z = torch.randint(0, K, (n,))
                z_one_hot = torch.nn.functional.one_hot(z, num_classes=K).float()
                prob_matrix = z_one_hot @ ps
                prob_matrix = prob_matrix @ z_one_hot.transpose(-1, -2)
                A = torch.distributions.Bernoulli(prob_matrix).sample()
                A = A.tril(diagonal=0) + A.tril(diagonal=-1).transpose(-1, -2)
                edge_index = pyg_utils.dense_to_sparse(A.unsqueeze(0))[0]

                # Generate features
                mu = torch.rand((K, 1)) * 3
                x = mu[z]

            elif self.graph_model == "full_SBM_Gaussian":
                # Randomly generate SBM parameters
                K = random.randint(10, 20)
                ps = torch.rand((K, K))  # random K x K probability matrix for SBM
                ps = ps.tril(diagonal=0) + ps.tril(diagonal=-1).transpose(-1, -2)

                # Generate graph
                z = torch.randint(0, K, (n,))
                z_one_hot = torch.nn.functional.one_hot(z, num_classes=K).float()
                prob_matrix = z_one_hot @ ps
                prob_matrix = prob_matrix @ z_one_hot.transpose(-1, -2)
                A = torch.distributions.Bernoulli(prob_matrix).sample()
                A = A.tril(diagonal=0) + A.tril(diagonal=-1).transpose(-1, -2)
                edge_index = pyg_utils.dense_to_sparse(A.unsqueeze(0))[0]
                # Generate features
                mu = torch.rand((K, 1)) * 3
                x = mu[z]

            if self.task == "triangle":
                y = torch.einsum("ij,jk,ki,id,jd,kd -> id", A, A, A, x, x, x) / (n**2)
                y = y.squeeze(-1)

            elif self.task == "degree":
                y = torch.einsum("ij,ji,id,jd -> id", A, A, x, x) / n
                y = y.squeeze(-1)

            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices
