import torch
import torch.nn as nn

class DeepSet(nn.Module):

    def __init__(self, in_channels: int=1, 
                 out_channels: int=1,
                 hidden_channels: int=50,
                 set_channels: int=50,
                 feature_extractor_num_layers: int=3,
                 regressor_num_layers: int=4,
                 normalized: bool=True,
                 **kwargs) -> None:
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
            torch.Tensor: B x N x out_channels
        """
        x = input # B x N x in_channels
        x = self.feature_extractor(x) # B x N x set_channels
        if self.normalized:
            x = x.mean(dim=1) # B x set_channels
        else:
            x = x.sum(dim=1) # B x set_channels
        x = self.regressor(x) # B x out_channels
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'Feature Exctractor=' + str(self.feature_extractor) \
            + '\n Set Feature' + str(self.regressor) + ')'