from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights
from rnd.models import ResNet18Classifier, MobileNetV2Classifier


def get_model(name, dense_units, dropout):
    if name == "resnet":
        weights = ResNet18_Weights.DEFAULT
        model = ResNet18Classifier(
            dense_units=dense_units, dropout_rate=dropout, weights=weights
        )
    elif name == "mobilenet":
        weights = MobileNet_V2_Weights.DEFAULT
        model = MobileNetV2Classifier(
            dense_units=dense_units, dropout_rate=dropout, weights=weights
        )
    else:
        raise ValueError(f"Unsupported model: {name}")

    return model, weights
