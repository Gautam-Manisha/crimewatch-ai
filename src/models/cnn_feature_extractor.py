import torch
import torch.nn as nn
from torchvision import models


class CNNFeatureExtractor(nn.Module):

    def __init__(self):

        super().__init__()

        # Load pretrained ResNet18
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-1]
        )

    def forward(self, x):

        """
        Input:
            x -> (batch, 3, 224, 224)

        Output:
            features -> (batch, 512)
        """

        features = self.feature_extractor(x)

        features = torch.flatten(
            features,
            start_dim=1
        )

        return features