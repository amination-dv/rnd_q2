import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class ResNet18Classifier(nn.Module):
    def __init__(
        self, dense_units=128, dropout_rate=0.5, weights=ResNet18_Weights.DEFAULT
    ):
        super(ResNet18Classifier, self).__init__()

        # Load pretrained ResNet18
        self.base_model = models.resnet18(weights=weights)

        # Freeze base layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Get the number of features in the original classifier
        in_features = self.base_model.fc.in_features

        # Replace the classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, dense_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_units, 1),
            nn.Sigmoid(),
        )

        # Replace original fc layer with our custom classifier
        self.base_model.fc = self.classifier

    def forward(self, x):
        return self.base_model(x)
