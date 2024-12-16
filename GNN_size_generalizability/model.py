from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor
from ign_layers import layer_2_to_2, layer_2_to_1, layer_2_to_1_anydim, layer_2_to_2_anydim

class GNN_layer(nn.Module):

    def __init__(
        self,
        A_in_channels: int,
        A_out_channels: int,
        x_in_channels: int,
        x_out_channels: int,
        reduced: bool = False,
    ) -> None:
        super().__init__()
        self.reduced = reduced

        # Initialize parameters
        self.A_l1, self.A_l2, self.A_l3, self.A_l4, self.A_l5 = nn.ModuleList(
            nn.Linear(A_in_channels, A_out_channels, bias=False) for _ in range(5)
        )
        self.A_l6, self.A_l7 = nn.ModuleList(
            [nn.Linear(x_in_channels, A_out_channels, bias=False) for _ in range(2)]
        )
        self.A_bias = nn.Parameter(torch.zeros(1, 1, 1, A_out_channels))

        self.X1_l1, self.X1_l2 = nn.ModuleList(
            [nn.Linear(x_in_channels, x_out_channels, bias=False) for _ in range(2)]
        )
        self.X1_l3, self.X1_l4, self.X1_l5, self.X1_l6 = nn.ModuleList(
            [nn.Linear(A_in_channels, x_out_channels, bias=False) for _ in range(4)]
        )
        self.X2_l1, self.X2_l2 = nn.ModuleList(
            [nn.Linear(x_in_channels, x_out_channels, bias=False) for _ in range(2)]
        )
        self.X2_l3, self.X2_l4, self.X2_l5, self.X2_l6 = nn.ModuleList(
            [nn.Linear(A_in_channels, x_out_channels, bias=False) for _ in range(4)]
        )
        self.X1_bias = nn.Parameter(torch.zeros(1, 1, x_out_channels))
        self.X2_bias = nn.Parameter(torch.zeros(1, 1, x_out_channels))
        self.out_transform = nn.Linear(A_out_channels * x_out_channels, x_out_channels)

    def forward(self, A: Tensor, X: Tensor) -> Tensor:
        """
        Args:
            A (Tensor): adjacency matrix, shape N, A_in, n, n
            X (Tensor): graph signal, shape N, n, x_in

        Returns:
            Tensors: (A,X) shape N, A_out, n, n, and N, n, x_out
        """
        if A.dim() == 3:
            A = A.unsqueeze(dim=1)
        assert A.dim() == 4 and A.shape[-1] == A.shape[-2], "A must be of shape  N, A_in, n, n"
        if X.dim() == 2:
            X = X.unsqueeze(dim=-1)
        assert (
            X.dim() == 3 and X.shape[0] == A.shape[0] and X.shape[-2] == A.shape[-1]
        ), "X must be of shape N, n, x_in"
        n = A.shape[-1]  # extract dimension
        diag_part = (
            torch.diagonal(A, dim1=-2, dim2=-1).unsqueeze(-1).permute(0, 2, 3, 1)
        )  # N, n, 1, A_in
        mean_diag_part = torch.mean(diag_part, dim=1).unsqueeze(dim=1)  # N, 1, 1, A_in
        mean_of_cols = torch.mean(A, dim=-1).unsqueeze(dim=-1).permute(0, 2, 3, 1)  # N, n, 1, A_in
        mean_all = torch.mean(mean_of_cols, dim=1).unsqueeze(dim=1)  # N, 1, 1, A_in
        mean_X = torch.mean(X, dim=-2).unsqueeze(dim=-2)  # N, 1, x_in

        A_transform = self.A_l1(A.permute(0, 2, 3, 1))  # N, n, n, A_out
        A_transform += self.A_l2(mean_all)  # N, 1, 1, A_out
        A_transform += self.A_l3((mean_of_cols + mean_of_cols.transpose(-2, -3)))  # N, n, n, A_out
        A_transform += self.A_l6(X.unsqueeze(dim=-2)) + self.A_l6(
            X.unsqueeze(dim=-3)
        )  # N, n, n, A_out
        A_transform += self.A_l7(mean_X.unsqueeze(1))  # N, 1, 1, A_out
        A_transform += self.A_bias
        if not self.reduced:
            A_transform += self.A_l4(mean_diag_part)  # N, 1, 1, A_out
            A_transform += self.A_l5(diag_part + diag_part.transpose(-2, -3))  # N, n, n, A_out

        X1_transform = self.X1_l1(X)  # N, n, x_out
        X1_transform += self.X1_l2(mean_X)  # N, 1, x_out
        X1_transform += self.X1_l3(mean_of_cols.squeeze(dim=-2))  # N, n, x_out
        X1_transform += self.X1_l6(mean_all.squeeze(dim=1))  # N, 1, x_out
        X1_transform += self.X1_bias  # 1, 1, x_out
        if not self.reduced:
            X1_transform += self.X1_l4(diag_part.squeeze(dim=-2))  # N, n, x_out
            X1_transform += self.X1_l5(mean_diag_part.squeeze(dim=-2))  # N, 1, x_out

        X2_transform = self.X2_l1(X)
        X2_transform += self.X2_l2(mean_X)
        X2_transform += self.X2_l3(mean_of_cols.squeeze(dim=-2))
        X2_transform += self.X2_l6(mean_all.squeeze(dim=1))
        X2_transform += self.X2_bias
        if not self.reduced:
            X2_transform += self.X2_l4(diag_part.squeeze(dim=-2))
            X2_transform += self.X2_l5(mean_diag_part.squeeze(dim=-2))

        out = torch.einsum("nijs, njt -> nist", A_transform, X2_transform) / n  # N, n, A_out, x_out
        out = out.reshape(out.shape[0], out.shape[1], -1)  # N, n, A_out * x_out
        out = self.out_transform(out)  # N, n, x_out
        out = out + X1_transform  # N, n, x_out
        return A_transform.permute(0, 3, 1, 2), out


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
        self.model = model
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        if self.model == "simple":
            if self.num_layers == 1:
                self.layers.append(GNNSimple_layer(self.in_channels, self.out_channels))
            else:
                self.layers.append(GNNSimple_layer(self.in_channels, self.hidden_channels))
                for _ in range(self.num_layers - 2):
                    self.layers.append(self.act)
                    self.layers.append(GNNSimple_layer(self.hidden_channels, self.hidden_channels))
                self.layers.append(self.act)
                self.layers.append(GNNSimple_layer(self.hidden_channels, self.out_channels))
        if self.model == "reduced" or self.model == "unreduced":
            reduced = True if self.model == "reduced" else False
            if self.num_layers == 1:
                self.layers.append(
                    GNN_layer(
                        A_in_channels=1,
                        A_out_channels=self.hidden_channels,
                        x_in_channels=self.in_channels,
                        x_out_channels=self.out_channels,
                        reduced=reduced,
                    )
                )
            else:
                self.layers.append(
                    GNN_layer(
                        1, self.hidden_channels, self.in_channels, self.hidden_channels, reduced
                    )
                )
                for _ in range(self.num_layers - 2):
                    self.layers.append(self.act)
                    self.layers.append(
                        GNN_layer(
                            self.hidden_channels,
                            self.hidden_channels,
                            self.hidden_channels,
                            self.hidden_channels,
                            reduced,
                        )
                    )
                self.layers.append(self.act)
                self.layers.append(
                    GNN_layer(
                        self.hidden_channels,
                        self.hidden_channels,
                        self.hidden_channels,
                        self.out_channels,
                        reduced,
                    )
                )

        if self.model == "ign":
            if self.num_layers == 1:
                self.layers.append(layer_2_to_1(self.in_channels + 1, self.out_channels))
            else:
                self.layers.append(layer_2_to_2(self.in_channels + 1, self.hidden_channels))
                for _ in range(self.num_layers - 2):
                    self.layers.append(self.act)
                    self.layers.append(layer_2_to_2(self.hidden_channels, self.hidden_channels))
                self.layers.append(self.act)
                self.layers.append(layer_2_to_1(self.hidden_channels, self.out_channels))

        if self.model == "ign_anydim":
            if self.num_layers == 1:
                self.layers.append(layer_2_to_1_anydim(self.in_channels + 1, self.out_channels))
            else:
                self.layers.append(layer_2_to_2_anydim(self.in_channels + 1, self.hidden_channels))
                for _ in range(self.num_layers - 2):
                    self.layers.append(self.act)
                    self.layers.append(
                        layer_2_to_2_anydim(self.hidden_channels, self.hidden_channels)
                    )
                self.layers.append(self.act)
                self.layers.append(layer_2_to_1_anydim(self.hidden_channels, self.out_channels))

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

        if self.model in ["ign", "ign_anydim"]:
            if X.dim() == 2:
                X = X.reshape(A.shape[0], A.shape[1], -1)
            X = torch.diag_embed(X.transpose(-1, -2))  # N x d x n x n
            A = A.unsqueeze(dim=1)  # N x 1 x n x n
            X = torch.cat([A, X], dim=1)  # N x (d+1) x n x n
            for layer in self.layers:
                X = layer(X)
            return X.transpose(-1, -2)
