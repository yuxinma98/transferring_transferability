import torch
import torch.nn as nn
from torchmetrics.functional.pairwise import pairwise_euclidean_distance
from torch_geometric.nn.models import MLP

from Anydim_transferability.DeepSet.model import DeepSet


class SiameseRegressor(torch.nn.Module):
    def __init__(self, params):
        super(SiameseRegressor, self).__init__()
        self.params = params
        if params["model_name"] == "SVD-DeepSet":
            self.SM = DeepSet(
                in_channels=3,
                out_channels=params["out_dim"],
                hidden_channels=params["hid_dim"],
                set_channels=params["hid_dim"] * 2,
                feature_extractor_num_layers=3,
                regressor_num_layers=2,
                normalization="sum",
            )
        elif params["model_name"] == "SVD-Normalized DeepSet":
            self.SM = DeepSet(
                in_channels=3,
                out_channels=params["out_dim"],
                hidden_channels=params["hid_dim"],
                set_channels=params["hid_dim"] * 2,
                feature_extractor_num_layers=3,
                regressor_num_layers=2,
                normalization="mean",
            )
        elif params["model_name"] == "DS-CI (Normalized)":
            self.SM = ScalarModel(**params)
        elif params["model_name"] == "OI-DS (Normalized)":
            self.SM = ScalarModel_KMeans(**params)
        self.metric = nn.Linear(params["out_dim"], params["out_dim"], bias=False)
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


class ScalarModel_KMeans(torch.nn.Module):
    def __init__(self, hid_dim, out_dim, dropout=0, input_dim=9, input_off=3, **kwargs):
        """
        input_dim = d^2
        input_off = d
        """
        super(ScalarModel_KMeans, self).__init__()
        # MLP for k-means gram matrix, C(Y) C(Y)^T
        self.MLP_km = MLP([input_dim, input_dim * 2, hid_dim], dropout=[dropout] * 2)
        # Deepset for k-means features, C(Y) Y (action on the n points in R^d)
        self.deepset_o = DeepSet(
            in_channels=input_off,
            out_channels=hid_dim,
            hidden_channels=hid_dim,
            set_channels=hid_dim * 2,
            feature_extractor_num_layers=3,
            regressor_num_layers=2,
            normalization="mean",
        )
        # MLP for (MLP_km(f_km), Deepset(f_d))
        self.MLP_out = MLP([hid_dim * 2, hid_dim, out_dim], dropout=[dropout] * 2)
        self.out_dim = out_dim

    def reset_parameters(self):
        for net in [self.MLP_km, self.deepset_o, self.MLP_out]:
            net.reset_parameters()

    def forward(self, data):
        out_d = self.MLP_km(data.f_d)  # bs x hid_dim
        out_o = self.deepset_o(data.f_o)  # bs x hid_dim
        # concat and output final embedding
        out = self.MLP_out(torch.concat([out_d, out_o], dim=-1))  # bs x hid_dim*2 -> bs x out_dim
        return out


class ScalarModel(torch.nn.Module):
    def __init__(self, hid_dim, out_dim, dropout=0, input_dim=1, input_off=1, **kwargs):
        """
        For all off-diag scalars, input_off = 1
        """
        super(ScalarModel, self).__init__()
        # Deepset for diagonal scalars
        self.deepset_d = DeepSet(
            in_channels=input_dim,
            out_channels=hid_dim,
            hidden_channels=hid_dim,
            set_channels=hid_dim * 2,
            feature_extractor_num_layers=3,
            regressor_num_layers=2,
            normalization="mean",
        )
        # Deepset for off-diagonal scalars
        self.deepset_o = DeepSet(
            in_channels=input_off,
            out_channels=hid_dim,
            hidden_channels=hid_dim,
            set_channels=hid_dim * 2,
            feature_extractor_num_layers=3,
            regressor_num_layers=2,
            normalization="mean",
        )
        # MLP_s for f_star
        self.MLP_s = MLP([input_dim, hid_dim, hid_dim], dropout=[dropout] * 2, norm=None)
        # MLP for (Deepset(f_d), Deepset(f_o), f_star)
        self.MLP_out = MLP([hid_dim * 3, hid_dim, out_dim], dropout=[dropout] * 2, norm=None)
        self.out_dim = out_dim

    def reset_parameters(self):
        for net in [self.deepset_d, self.deepset_o, self.MLP_s, self.MLP_out]:
            # for net in [self.deepset_d, self.deepset_o, self.MLP_out]:
            net.reset_parameters()

    def forward(self, data):
        out_d = self.deepset_d(data.f_d)  # bs x hid_dim
        out_o = self.deepset_o(data.f_o)  # bs x hid_dim
        out_star = self.MLP_s(data.f_star)  # bs x hid_dim
        # concat and output final embedding
        out = self.MLP_out(
            torch.concat([out_d, out_o, out_star], dim=-1)
        )  # bs x hid_dim*3 -> bs x out_dim
        return out
