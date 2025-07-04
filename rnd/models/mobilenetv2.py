import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights


class MobileNetV2Classifier(nn.Module):
    def __init__(
        self, dense_units=128, dropout_rate=0.5, weights=MobileNet_V2_Weights.DEFAULT
    ):
        super(MobileNetV2Classifier, self).__init__()

        # Load MobileNetV2 with specified weights
        self.base_model = models.mobilenet_v2(weights=weights)

        # Freeze base layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Get number of features from the last layer of MobileNetV2
        in_features = self.base_model.classifier[1].in_features

        # Replace the classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.classifier(x)
        return x
