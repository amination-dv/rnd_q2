from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights
from rnd.models import ResNet18Classifier, MobileNetV2Classifier
from rnd.models.unet import ShallowUNetAT
from rnd.models.tcn import SmallTCN


def get_model(name, dense_units=None, dropout=None, num_tracks=20):
    weights = None
    if name == "resnet":
        weights = ResNet18_Weights.DEFAULT
        model = ResNet18Classifier(
            dense_units=dense_units,
            dropout_rate=dropout,
            weights=weights,
            num_classes=num_tracks,
        )
    elif name == "mobilenet":
        weights = MobileNet_V2_Weights.DEFAULT
        model = MobileNetV2Classifier(
            dense_units=dense_units, dropout_rate=dropout, weights=weights
        )
    elif name == "unet":
        model = ShallowUNetAT()
        
    elif name == "tcn":
        model = SmallTCN(hidden=64, blocks=5) 
    else:
        raise ValueError(f"Unsupported model: {name}")

    return model, weights
