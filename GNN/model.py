from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor
from ign_layers import layer_2_to_2, layer_2_to_1

class GNN_layer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduced: bool = False) -> None:
        super().__init__()
        self.in_channels = in_channels #D1
        self.out_channels = out_channels #D2
        self.reduced = reduced

        # Initialize parameters
        self.A_coeffs = nn.Parameter(torch.randn(5), requires_grad = True)
        self.A_l1, self.A_l2 = nn.ModuleList([nn.Linear(in_channels, 1, bias=False) for _ in range(2)])

        self.X1_l1, self.X1_l2 = nn.ModuleList([nn.Linear(in_channels, out_channels, bias=False) for _ in range(2)])
        self.X1_l3, self.X1_l4, self.X1_l5, self.X1_l6 = nn.ModuleList([nn.Linear(1, out_channels, bias=False) for _ in range(4)])
        self.X2_l1, self.X2_l2 = nn.ModuleList([nn.Linear(in_channels, out_channels, bias=False) for _ in range(2)])
        self.X2_l3, self.X2_l4, self.X2_l5, self.X2_l6 = nn.ModuleList([nn.Linear(1, out_channels, bias=False) for _ in range(4)])

    def forward(self, A: Tensor, X: Tensor) -> Tensor:
        """
        Args:
            A (Tensor): adjacency matrix, shape N x n x n
            X (Tensor): graph signal, shape N x n x D1

        Returns:
            Tensor: output signal, shape N x n x D2
        """
        assert A.dim() == 3 and A.shape[1] == A.shape[2], "A must be of shape N x n x n"
        if X.dim() == 2:
            X = X.reshape(A.shape[0], A.shape[1], -1)
        assert (
            X.dim() == 3 and X.shape[0] == A.shape[0] and X.shape[1] == A.shape[1]
        ), "X must be of shape N x n x D1"
        n = A.shape[-1]  # extract dimension
        A = A.unsqueeze(dim=1) # N x 1 x n x n
        diag_part = torch.diagonal(A, dim1=-2, dim2=-1)   # N x 1 x n
        mean_diag_part = torch.mean(diag_part, dim=-1).unsqueeze(dim=-1)  # N x 1 x 1
        mean_of_cols = torch.mean(A, dim=-1)  # N x 1 x n
        mean_all = torch.mean(mean_of_cols, dim=-1).unsqueeze(dim=-1)  # N x 1 x 1
        mean_X = torch.mean(X, dim=-2).unsqueeze(dim=-2) #N x 1 x D1

        A_transform = self.A_coeffs[0] * A.squeeze(dim=1)
        A_transform += self.A_coeffs[1] * mean_all 
        A_transform += self.A_coeffs[3] * (mean_of_cols + mean_of_cols.transpose(-1,-2))
        A_transform += (self.A_l1(X.unsqueeze(dim=-2)) + self.A_l1(X.unsqueeze(dim=-3))).squeeze(dim=-1)
        A_transform += self.A_l2(mean_X).expand(-1,n,n)
        if not self.reduced:
            A_transform += self.A_coeffs[2] * mean_diag_part
            A_transform += self.A_coeffs[4] * (diag_part + diag_part.transpose(-1,-2))

        X1_transform = self.X1_l1(X)
        X1_transform += self.X1_l2(mean_X)
        X1_transform += self.X1_l3(mean_of_cols.squeeze(dim=-2).unsqueeze(dim=-1))
        X1_transform += self.X1_l6(mean_all)
        if not self.reduced:
            X1_transform += self.X1_l4(diag_part.squeeze(dim=-2).unsqueeze(dim=-1))
            X1_transform += self.X1_l5(mean_diag_part)

        X2_transform = self.X2_l1(X)
        X2_transform += self.X2_l2(mean_X)
        X2_transform += self.X2_l3(mean_of_cols.squeeze(dim=-2).unsqueeze(dim=-1))
        X2_transform += self.X2_l6(mean_all)
        if not self.reduced:
            X2_transform += self.X2_l4(diag_part.squeeze(dim=-2).unsqueeze(dim=-1))
            X2_transform += self.X2_l5(mean_diag_part)

        out = A_transform.matmul(X2_transform) / n + X1_transform
        return A_transform, out


class GNNSimple_layer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, reduced=False) -> None:
        super().__init__()
        self.in_channels = in_channels  # D1
        self.out_channels = out_channels  # D2

        # Initialize parameters
        self.X1_l, self.X2_l = nn.ModuleList(
            [nn.Linear(in_channels, out_channels, bias=False) for _ in range(2)]
        )

    def forward(self, A: Tensor, X: Tensor) -> Tensor:
        """
        Args:
            A (Tensor): adjacency matrix, shape N x n x n
            X (Tensor): graph signal, shape N x n x D1

        Returns:
            Tensor: output signal, shape N x n x D2
        """
        assert A.dim() == 3 and A.shape[1] == A.shape[2], "A must be of shape N x n x n"
        if X.dim() == 2:
            X = X.reshape(A.shape[0], A.shape[1], -1)
        assert (
            X.dim() == 3 and X.shape[0] == A.shape[0] and X.shape[1] == A.shape[1]
        ), "X must be of shape N x n x D1"
        n = A.shape[-1]  # extract dimension

        X1_transform = self.X1_l(X)
        X2_transform = self.X2_l(X)

        out = A.matmul(X2_transform) / n + X1_transform
        return A, out


class GNN(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        out_channels: int,
        act: Callable = nn.ReLU(),
        model: str = "simple",  # choices in ["simple", "reduced", "unreduced", "ign"]
        **kwargs
    ) -> None:
        super(GNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.act = act
        assert model in ["simple", "reduced", "unreduced", "ign"]
        self.model = model
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        if self.model in ["simple", "reduced", "unreduced"]:
            if self.model == "simple":
                nn_layer = GNNSimple_layer
                reduced = None
            elif self.model == "reduced":
                nn_layer = GNN_layer
                reduced = True
            elif self.model == "unreduced":
                nn_layer = GNN_layer
                reduced = False

            if self.num_layers == 1:
                self.layers.append(nn_layer(self.in_channels, self.out_channels, reduced))
            else:
                self.layers.append(nn_layer(self.in_channels, self.hidden_channels, reduced))
                for _ in range(self.num_layers - 2):
                    self.layers.append(self.act)
                    self.layers.append(
                        nn_layer(self.hidden_channels, self.hidden_channels, reduced)
                    )
                self.layers.append(self.act)
                self.layers.append(nn_layer(self.hidden_channels, self.out_channels, reduced))

        if self.model == "ign":
            self.linear = nn.Linear(self.in_channels, 3)
            if self.num_layers == 1:
                self.layers.append(layer_2_to_1(4, self.out_channels))
            else:
                self.layers.append(layer_2_to_2(4, self.hidden_channels))
                for _ in range(self.num_layers - 2):
                    self.layers.append(self.act)
                    self.layers.append(layer_2_to_2(self.hidden_channels, self.hidden_channels))
                self.layers.append(self.act)
                self.layers.append(layer_2_to_1(self.hidden_channels, self.out_channels))

    def forward(self, adj: Tensor, feature: Tensor) -> Tensor:
        """
        Args:
            adj (Tensor): adjacency matrix, shape N x n x n
            feature (Tensor): graph signal, shape N x n x D1

        Returns:
            Tensor: output signal, shape N x n x D2
        """
        A = adj.clone()
        X = feature.clone()
        if self.model in ["simple", "reduced", "unreduced"]:
            for layer in self.layers:
                if isinstance(layer, GNN_layer) or isinstance(layer, GNNSimple_layer):
                    A, X = layer(A, X)
                else:
                    X = layer(X)
            return X

        if self.model == "ign":
            if X.dim() == 2:
                X = X.reshape(A.shape[0], A.shape[1], -1)
            X = self.linear(X)  # N x n x 3
            X = torch.diag_embed(X.transpose(-1, -2))  # N x 3 x n x n
            A = A.unsqueeze(dim=1)  # N x 1 x n x n
            X = torch.cat([A, X], dim=1)  # N x 4 x n x n
            for layer in self.layers:
                X = layer(X)
            return X.transpose(-1, -2)
