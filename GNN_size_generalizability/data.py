import torch
import random
from torch_geometric.data import Data, InMemoryDataset
import torch_geometric.utils as pyg_utils


class DegreeDataset(InMemoryDataset):
    def __init__(self, root, N, n, d):
        self.N = N
        self.n = n
        self.d = d
        super(DegreeDataset, self).__init__(root=root, transform=None, pre_transform=None)
        self.data, self.slices = self.process()

    @property
    def processed_file_names(self):
        return [f"data_{self.N}_{self.n}.pt"]

    def process(self):
        data_list = []

        for i in range(self.N):
            if isinstance(self.n, int):
                n = self.n
            else:
                n = random.randint(self.n[0], self.n[1])
            edge_index = pyg_utils.erdos_renyi_graph(n, 1 / 2)
            x = torch.randn((n, self.d)).unsqueeze(0)  # Random node features
            data = Data(x=x, edge_index=edge_index)
            data.A = pyg_utils.to_dense_adj(edge_index)
            data.y = pyg_utils.degree(edge_index[0]).unsqueeze(0) / n
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        return data, slices
