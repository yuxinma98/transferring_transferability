import pickle
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as pyg_DataLoader
from sklearn.cluster import KMeans
from torch_geometric.data import Data
import pytorch_lightning as pl
import torch
import numpy as np


class GWLBDataModule(pl.LightningDataModule):

    def __init__(self, fname, model_name):
        super(GWLBDataModule, self).__init__()
        self.fname = fname
        self.model_name = model_name

    def prepare_data(self):
        xtrain_s, ytrain_s, xtest_s, ytest_s, kernel_train, kernel_test = pickle.load(
            open(
                self.fname,
                "rb",
            )
        )
        xtrain_s = torch.from_numpy(np.array(xtrain_s)).float()
        xtest_s = torch.from_numpy(np.array(xtest_s)).float()
        ytrain_s = torch.from_numpy(np.array(ytrain_s)).long().squeeze()
        ytest_s = torch.from_numpy(np.array(ytest_s)).long().squeeze()
        X1_train, X2_train = xtrain_s[ytrain_s == 2], xtrain_s[ytrain_s == 7]
        X1_test, X2_test = xtest_s[ytest_s == 2], xtest_s[ytest_s == 7]
        self.dist_true_train = torch.from_numpy(np.array(kernel_train)).unsqueeze(1)
        self.dist_true_test = torch.from_numpy(np.array(kernel_test)).unsqueeze(1)

        if self.model_name == "SVD-DeepSet" or self.model_name == "SVD-Normalized DeepSet":
            X1_train, X2_train = self._svd(X1_train), self._svd(
                X2_train
            )  # num_pointclouds x num_points x 3
            X1_test, X2_test = self._svd(X1_test), self._svd(X2_test)
            self.X_train = torch.stack([X1_train, X2_train], dim=0)
            self.X_test = torch.stack([X1_test, X2_test], dim=0)

        elif self.model_name == "DS-CI (Normalized)":
            self.X_train = [self._get_fs(X1_train), self._get_fs(X2_train)]
            self.X_test = [self._get_fs(X1_test), self._get_fs(X2_test)]

        elif self.model_name == "OI-DS (Normalized)":
            X1_train = [self._get_Kmeans(X1_train[i]) for i in range(X1_train.shape[0])]
            X2_train = [self._get_Kmeans(X2_train[i]) for i in range(X2_train.shape[0])]
            X1_test = [self._get_Kmeans(X1_test[i]) for i in range(X1_test.shape[0])]
            X2_test = [self._get_Kmeans(X2_test[i]) for i in range(X2_test.shape[0])]
            self.X_train = X1_train + X2_train
            self.X_test = X1_test + X2_test

    def _svd(self, X: torch.Tensor) -> torch.Tensor:
        """
        Suppose SVD of X is X=UDV^T, D is ordered by absolute value.
        Output S=UD=XV
        Args:
            X (torch.Tensor): num_pointclouds x num_points x 3

        Returns:
            torch.Tensor: num_pointclouds x num_points x 3
        """
        # X:
        S = torch.matmul(X.transpose(1, 2), X)  # num_pointclouds x 3 x 3
        U, _, _ = torch.svd(S)  # num_pointclouds x 3 x 3
        S = torch.matmul(X, U)  # num_pointclouds x num_points x 3
        return S

    def _get_fs(self, X: torch.Tensor) -> Data:
        """
        Generate the invariant features f_d, f_o, f_star
        Args:
            X (torch.Tensor): num_pointclouds x num_points x 3
        Returns:
            Data: Data object with f_d, f_o, f_star
        """
        num_pointclouds = X.shape[0]
        n = X.shape[1]
        data = Data()
        Gram = torch.FloatTensor(torch.matmul(X, X.transpose(1, 2)))  # num_pointclouds x n x n
        off_mask = torch.triu(torch.ones(n, n)) == 1

        data.f_d = torch.diagonal(Gram, 0, dim1=1, dim2=2).unsqueeze(-1)  # num_pointclouds x n x 1
        data.f_o = Gram[:, off_mask].unsqueeze(-1)  # num_pointclouds x n(n-1)/2 x 1
        data.f_star = (
            (data.f_d * (torch.sum(Gram, dim=2, keepdim=True) - data.f_d)).sum(dim=1)
        ) / (
            n * (n - 1)
        )  # num_pointclouds x 1
        return data

    def _get_Kmeans(self, X, k=3):
        """
        Implement eqn 6: replace f_d as the self-dots of the K-means dots
        input: X (n by 3), k: number of k-means clusters
        output: Data (class) storing  feature matrix
        f_d is the flattened feature of the gram matrix of K-means centroids
        f_o is the set of dot products of each point to the K-means centroids
        """
        n = X.shape[0]
        data = Data()

        kmeans = KMeans(n_clusters=k, random_state=0, init="k-means++", n_init=1).fit(X)
        K = kmeans.cluster_centers_  # (num_centroids, 3)
        # sort Kmeans centroid by norm
        indexlist = np.argsort(np.linalg.norm(K, axis=1))
        K = K[indexlist, :]
        data.f_o = torch.FloatTensor(X @ K.T)  # shape(n, d)
        Gram_k = torch.FloatTensor(K @ K.T)  # shape (d, d)
        data.f_d = Gram_k.reshape(1, k * k)  # shape (d^2,)
        return data

    def train_dataloader(self):
        if self.model_name == "SVD-DeepSet" or self.model_name == "SVD-Normalized DeepSet":
            return DataLoader(self.X_train, batch_size=len(self.X_train), shuffle=False)
        elif self.model_name == "DS-CI (Normalized)" or self.model_name == "OI-DS (Normalized)":
            return pyg_DataLoader(self.X_train, batch_size=2, shuffle=False)

    def val_dataloader(self):
        if self.model_name == "SVD-DeepSet" or self.model_name == "SVD-Normalized DeepSet":
            return DataLoader(self.X_train, batch_size=len(self.X_train), shuffle=False)
        elif self.model_name == "DS-CI (Normalized)" or self.model_name == "OI-DS (Normalized)":
            return pyg_DataLoader(self.X_train, batch_size=2, shuffle=False)

    def test_dataloader(self):
        if self.model_name == "SVD-DeepSet" or self.model_name == "SVD-Normalized DeepSet":
            return DataLoader(self.X_test, batch_size=len(self.X_test), shuffle=False)
        elif self.model_name == "DS-CI (Normalized)" or self.model_name == "OI-DS (Normalized)":
            return pyg_DataLoader(self.X_test, batch_size=2, shuffle=False)
