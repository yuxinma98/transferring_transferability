import h5py
import os
import numpy as np
import torch
from typing import Union
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import pickle
from torch.utils.data import random_split

class PopStatsDataset(Dataset):
    def __init__(self, fname: Union[str, os.PathLike]) -> None:
        self.fname = fname
        
        # Load data from Matlab
        with h5py.File(fname, 'r') as f:
            self.L = torch.tensor(f['L'][()], dtype=torch.int32).item()  # number of sets
            self.N = torch.tensor(f['N'][()], dtype=torch.int32).item()  # number of elements in set
            self.t = torch.tensor(np.squeeze(f['X_parameter'][()]), dtype=torch.float32) # parametrization of distribution. e.g. rotation angles for task 1
            self.X = torch.tensor(f['X'][()], dtype=torch.float32)
            self.d = self.X.shape[0]
            self.X = self.X.transpose(0, 1).reshape(-1, self.N, self.d)
            self.y = torch.tensor(f['Y'][()], dtype=torch.float32).reshape(-1, 1)
        
    def __len__(self):
        return self.L
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PopStatsDataModule(LightningDataModule):
    def __init__(self, data_dir: Union[str, os.PathLike], task_id: int, batch_size: int, training_size: int, **kwargs) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.task_id = task_id
        self.batch_size = batch_size
        self.size = training_size

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = PopStatsDataset(os.path.join(self.data_dir, f'task{self.task_id}/train_{self.size}.mat'))
        self.val_dataset = PopStatsDataset(os.path.join(self.data_dir, f'task{self.task_id}/val_{self.size}.mat'))
        self.test_dataset = PopStatsDataset(os.path.join(self.data_dir, f'task{self.task_id}/test_{self.size}.mat'))
        self.truth = PopStatsDataset(os.path.join(self.data_dir, f'task{self.task_id}/truth.mat'))
        self.d = self.train_dataset.d

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


class HausdorffDataset(Dataset):
    def __init__(self, data_dir: Union[str, os.PathLike], N: int, n: int) -> None:
        fname = os.path.join(data_dir, f"circle_{N}_{n}.pkl")
        self.N = N
        self.n = n
        try:
            self.X, self.y = pickle.load(open(fname, "rb"))
        except FileNotFoundError:
            center = torch.randn(N, 2)
            radius = torch.rand(N, 1)
            theta = torch.rand(N, n) * 2 * np.pi
            self.X = torch.stack(
                [
                    center[:, 0].unsqueeze(-1) + radius * torch.cos(theta),
                    center[:, 1].unsqueeze(-1) + radius * torch.sin(theta),
                ],
                dim=2,
            )  # N x n x 2
            self.y = torch.max(torch.norm(self.X, dim=2), dim=1).values
            self.y = self.y.unsqueeze(-1)  # N x 1

            with open(fname, "wb") as f:
                pickle.dump((self.X, self.y), f)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class HausdorffDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Union[str, os.PathLike],
        N: int,
        n: int,
        batch_size: int,
        **kwargs,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.N = N
        self.n = n
        self.batch_size = batch_size

    def prepare_data(self):
        self.dataset = HausdorffDataset(
            data_dir=self.data_dir,
            N=self.N,
            n=self.n,
        )
        self.d = 2

    def setup(self, stage=None):
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [int(0.8 * self.N), int(0.1 * self.N), int(0.1 * self.N)],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
