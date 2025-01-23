import torch
import torch.nn as nn
from torchmetrics.functional.pairwise import pairwise_euclidean_distance


class SiameseRegressor(torch.nn.Module):
    def __init__(self, params):
        super(SiameseRegressor, self).__init__()
        self.SM = DeepSet(in_channels=3, **params)
        self.metric = nn.Linear(params["out_channels"], params["out_channels"], bias=False)
        self.linear = nn.Linear(1, 1)

    def forward(self, x1, x2):
        emb1 = self.SM(x1)  # num_pointclouds x out_dim
        emb2 = self.SM(x2)  # num_pointclouds x out_dim
        emb1 = self.metric(emb1)  # project
        emb2 = self.metric(emb2)  # project
        dist_mat = pairwise_euclidean_distance(emb1, emb2)  # num_pointclouds x num_pointclouds
        dist_vec = dist_mat.flatten().unsqueeze(1)
        dist_pred = self.linear(dist_vec)
        return dist_pred


class DeepSet(nn.Module):

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 50,
        set_channels: int = 50,
        feature_extractor_num_layers: int = 3,
        regressor_num_layers: int = 4,
        normalized: bool = True,
        **kwargs,
    ) -> None:
        super(DeepSet, self).__init__()
        self.normalized = normalized

        # Feature extractor
        feature_extractor_layers = [nn.Linear(in_channels, hidden_channels)]
        for _ in range(feature_extractor_num_layers - 2):
            feature_extractor_layers.append(nn.ELU(inplace=True))
            feature_extractor_layers.append(nn.Linear(hidden_channels, hidden_channels))
        feature_extractor_layers.append(nn.ELU(inplace=True))
        feature_extractor_layers.append(nn.Linear(hidden_channels, set_channels))
        self.feature_extractor = nn.Sequential(*feature_extractor_layers)

        # Regressor
        if regressor_num_layers == 1:
            self.regressor = nn.Sequential(nn.Linear(set_channels, out_channels))
        else:
            regressor_layers = [nn.Linear(set_channels, hidden_channels)]
            for _ in range(regressor_num_layers - 2):
                regressor_layers.append(nn.ELU(inplace=True))
                regressor_layers.append(nn.Linear(hidden_channels, hidden_channels))
            regressor_layers.append(nn.ELU(inplace=True))
            regressor_layers.append(nn.Linear(hidden_channels, out_channels))
            self.regressor = nn.Sequential(*regressor_layers)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """

        Args:
            input (torch.Tensor): B x N x in_channels

        Returns:
            torch.Tensor: B x out_channels
        """
        x = input  # B x N x in_channels
        x = self.feature_extractor(x)  # B x N x set_channels
        if self.normalized:
            x = x.mean(dim=1)  # B x set_channels
        else:
            x = x.sum(dim=1)  # B x set_channels
        x = self.regressor(x)  # B x out_channels
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "Feature Exctractor="
            + str(self.feature_extractor)
            + "\n Set Feature"
            + str(self.regressor)
            + ")"
        )
