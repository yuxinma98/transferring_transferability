from typing import Callable
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

class GNN_layer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels #D1
        self.out_channels = out_channels #D2

        self.A_num_params = 5 + 2 * in_channels
        self.X_num_params = 4 + 2 * in_channels

        # Initialize parameters
        self.A_coeffs = nn.Parameter(torch.randn(self.A_num_params), requires_grad = True)
        self.X_coeffs_1 = nn.Parameter(torch.randn(self.out_channels, self.X_num_params) * np.sqrt(2.0) / (self.out_channels + self.X_num_params), requires_grad = True)
        self.X_coeffs_2 = nn.Parameter(torch.randn(self.out_channels, self.X_num_params) * np.sqrt(2.0) / (self.out_channels + self.X_num_params), requires_grad = True)

    def forward(self, A: Tensor, X: Tensor) -> Tensor:
        """
        Args:
            A (Tensor): adjacency matrix, shape N x n x n
            X (Tensor): graph signal, shape N x n x D1

        Returns:
            Tensor: output signal, shape N x n x D2
        """
        n = A.shape[-1]  # extract dimension
        # Compute basis
        A = A.unsqueeze(dim=1) # N x 1 x n x n
        X = X.transpose(-2, -1) # N x D1 x n
        diag_part = torch.diagonal(A, dim1=-2, dim2=-1)   # N x 1 x n
        mean_diag_part = torch.mean(diag_part, dim=-1).unsqueeze(dim=-1)  # N x 1 x 1
        mean_of_cols = torch.mean(A, dim=-1)  # N x 1 x n
        mean_all = torch.mean(mean_of_cols, dim=-1).unsqueeze(dim=-1)  # N x 1 x 1
        mean_X = torch.mean(X, dim=-1).unsqueeze(dim=-1) #N x D1 x 1
        a1 = A
        a2 = mean_all.unsqueeze(dim=-1).expand(-1, -1, n, n)
        a3 = mean_diag_part.unsqueeze(dim=-1).expand(-1, -1, n, n)
        a4 = mean_of_cols.unsqueeze(dim=-1).expand(-1, -1, n, n) + mean_of_cols.unsqueeze(dim=-2).expand(-1, -1, n, n)
        a5 = diag_part.unsqueeze(dim=-1).expand(-1, -1, n, n) + diag_part.unsqueeze(dim=-2).expand(-1, -1, n, n)
        a6 = X.unsqueeze(dim=-1).expand(-1, -1, n, n) + X.unsqueeze(dim=-2).expand(-1, -1, n, n) # N x D1 x n x n
        a7 = mean_X.unsqueeze(dim=-1).expand(-1, -1, n, n) # N x D1 x n x n
        x1 = X # N x D1 x n
        x2 = mean_X.expand(-1, -1, n) # N x D1 x n
        x3 = mean_of_cols # N x 1 x n
        x4 = diag_part # N x 1 x n
        x5 = mean_diag_part.expand(-1, -1, n) # N x 1 x n
        x6 = mean_all.expand(-1, -1, n) # N x 1 x n

        # Multiply learnable coefficients
        out_a = torch.concat([a1, a2, a3, a4, a5, a6, a7], dim=1) # N x (5 + 2*D1) x n x n 
        A_transformed = torch.einsum('d,ndij->nij', self.A_coeffs, out_a) # N x n x n
        out_x = torch.concat([x1, x2, x3, x4, x5, x6], dim=1) # N x (4 + 2*D1) x n
        X_transformed_1 = torch.einsum('bd,ndi->nbi', self.X_coeffs_1, out_x) # N x D2 x n
        X_transformed_2 = torch.einsum('bd,ndi->nbi', self.X_coeffs_2, out_x) # N x D2 x n
        
        out = A_transformed.matmul(X_transformed_1.transpose(-1,-2)).transpose(-1,-2) / n + X_transformed_2 # N x D2 x n
        out = out.transpose(-2, -1) # N x n x D2
        return out

class GNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int,
                 act: Callable = nn.ReLU(), **kwargs) -> None:
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.act = act
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        if self.num_layers == 1:
            self.layers.append(GNN_layer(self.in_channels, self.out_channels))
        else:
            self.layers.append(GNN_layer(self.in_channels, self.hidden_channels))
            for _ in range(self.num_layers - 2):
                self.layers.append(self.act)
                self.layers.append(GNN_layer(self.hidden_channels, self.hidden_channels))
            self.layers.append(self.act)
            self.layers.append(GNN_layer(self.hidden_channels, self.out_channels))

    def forward(self, A: Tensor, X: Tensor) -> Tensor:
        """
        Args:
            A (Tensor): adjacency matrix, shape N x n x n
            X (Tensor): graph signal, shape N x n x D1

        Returns:
            Tensor: output signal, shape N x n x D2
        """
        for layer in self.layers:
            if isinstance(layer, GNN_layer):
                X = layer(A, X)
            else:
                X = layer(X)
        return X
