import torch
import random
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.utils as pyg_utils


class HomDensityDataset(InMemoryDataset):

    def __init__(self, root, N, n, d, graph_model, task, **kwargs):
        self.N = N
        self.n = n
        self.d = d
        self.graph_model = graph_model
        self.task = task
        if self.graph_model == "ER":
            self.p = [0.3, 0.7]
        elif self.graph_model == "SBM":
            self.ps = torch.tensor([[0.8, 0.15, 0.05], [0.15, 0.7, 0.25], [0.05, 0.25, 0.9]])
        elif self.graph_model == "Sociality":
            self.kernel = lambda x, y: 1 / (1 + torch.exp(-x * y))
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
                A = pyg_utils.to_dense_adj(edge_index)
            elif self.graph_model == "SBM":
                z = torch.randint(0, 3, (n,))
                z_one_hot = torch.nn.functional.one_hot(z, num_classes=3).float()
                prob_matrix = z_one_hot @ self.ps
                prob_matrix = prob_matrix @ z_one_hot.transpose(-1, -2)
                A = torch.distributions.Bernoulli(prob_matrix).sample()
                A = A.tril(diagonal=-1) + A.tril(diagonal=-1).transpose(-1, -2)
                A = A.unsqueeze(0)  # 1 x n x n
                edge_index = pyg_utils.dense_to_sparse(A)[0]
            elif self.graph_model == "Sociality":
                uniform_sampler = torch.distributions.Uniform(0, 1)
                z = uniform_sampler.sample((n,))
                # a = random.uniform(0.3, 0.7)
                prob_matrix = self.kernel(z.unsqueeze(1), z.unsqueeze(0))
                A = torch.distributions.Bernoulli(prob_matrix).sample()
                A = A.tril(diagonal=-1) + A.tril(diagonal=-1).transpose(-1, -2)
                A = A.unsqueeze(0)
                edge_index = pyg_utils.dense_to_sparse(A)[0]

            x = torch.randn((n, self.d)).unsqueeze(0)  # Random node features, 1 x n x d
            data = Data(x=x, edge_index=edge_index, A=A)

            if self.task == "degree":
                data.y = pyg_utils.degree(edge_index[0]).sum().unsqueeze(0) / (n**2)
            elif self.task == "triangle":
                A = A.squeeze()
                triangles = (A @ A @ A).diag().sum() / 6
                data.y = triangles / (n**3)  # 1 x n

            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices
