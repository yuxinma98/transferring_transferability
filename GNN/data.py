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
        n: int,
        graph_model: str,
        task: str,
        **kwargs,
    ):
        """
        Args:
            root (Union[str, os.PathLike]): directory to save the dataset
            N (int): Number of graphs
            n (int): Number of nodes
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
        if self.graph_model == "SBM":
            return self.process_SBM()
        elif self.graph_model == "full_random":
            return self.process_full_random()
        else:
            raise ValueError("Invalid graph model")

    def process_full_random(self):
        A = torch.rand((self.N, self.n, self.n))
        A = A.tril(diagonal=0) + A.tril(diagonal=-1).transpose(-1, -2)
        x = torch.rand((self.N, self.n, 1))
        data_list = []
        for i in range(self.N):  # process one-by-one to avoid memory issues
            Ai, xi = A[i], x[i]
            if self.task == "triangle":
                y = torch.einsum("ij,jk,ki,id,jd,kd -> id", Ai, Ai, Ai, xi, xi, xi) / (self.n**2)
            elif self.task == "degree":
                y = torch.einsum("ij,ji,id,jd -> id", Ai, Ai, xi, xi) / self.n
            data_list.append(Data(x=xi.unsqueeze(0), A=Ai.unsqueeze(0), y=y.squeeze(-1)))
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices

    def process_SBM(self):
        data_list = []
        for i in range(self.N):
            # Randomly generate SBM parameters
            K = random.randint(10, 20)
            ps = torch.rand((K, K))  # random K x K probability matrix for SBM
            ps = ps.tril(diagonal=0) + ps.tril(diagonal=-1).transpose(-1, -2)

            # Generate graph
            z = torch.randint(0, K, (self.n,))
            z_one_hot = torch.nn.functional.one_hot(z, num_classes=K).float()
            prob_matrix = z_one_hot @ ps
            prob_matrix = prob_matrix @ z_one_hot.transpose(-1, -2)
            A = torch.distributions.Bernoulli(prob_matrix).sample()
            A = A.tril(diagonal=0) + A.tril(diagonal=-1).transpose(-1, -2)
            # Generate features
            mu = torch.rand((K, 1))
            x = mu[z]
            if self.task == "triangle":
                y = torch.einsum("ij,jk,ki,id,jd,kd -> id", A, A, A, x, x, x) / (self.n**2)
            elif self.task == "degree":
                y = torch.einsum("ij,ji,id,jd -> id", A, A, x, x) / self.n

            data = Data(
                x=x.unsqueeze(0),
                edge_index=pyg_utils.dense_to_sparse(A)[0],
                A=A.unsqueeze(0),
                y=y.squeeze(-1),
            )
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices
