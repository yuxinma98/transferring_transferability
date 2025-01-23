import h5py
import os
import numpy as np
import torch
from typing import Union
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

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


class SubsampledDataset(Dataset):
    def __init__(self, X, n_samples, set_size):
        N, d = X.shape[-2], X.shape[-1]
        self.X = torch.zeros(n_samples, set_size, d)
        for i in range(n_samples):
            choices = torch.randint(0, N, (set_size,))
            self.X[i, :] = X[:, choices, :]

    def __len__(self):
        return self.X.shape[0]    
    
    def __getitem__(self, idx):
        return self.X[idx]