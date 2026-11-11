"""Models. Every model returns class scores (logits) for the 4 classes of D14."""
import inspect

from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class P1(nn.Module):
    """MobileNetV2 per frame, with the head of the reference paper (D21).

    Head: 3 x 3 convolution with 100 filters (7 x 7 to 5 x 5), max pooling 2 x 2 (to 2 x 2),
    flatten (400 values), then the dense layers. The paper gives no number of units.
    """

    def __init__(self, n_classes=4, dense_layers=5, units=128, pretrained=True):
        super().__init__()
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = mobilenet_v2(weights=weights).features  # output: 1280 x 7 x 7
        head = [nn.Conv2d(1280, 100, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten()]
        size = 400
        for _ in range(dense_layers):
            head += [nn.Linear(size, units), nn.ReLU()]
            size = units
        self.head = nn.Sequential(*head, nn.Linear(size, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


class B1(nn.Module):
    """Small CNN per frame, trained from scratch (D29).

    4 blocks of convolution, batch normalization, ReLU, and max pooling (224 x 224 to 14 x 14),
    then global average pooling and the output layer.
    """

    def __init__(self, n_classes=4):
        super().__init__()
        layers, size = [], 3
        for filters in (32, 64, 128, 256):
            layers += [nn.Conv2d(size, filters, 3, padding=1), nn.BatchNorm2d(filters), nn.ReLU(), nn.MaxPool2d(2)]
            size = filters
        self.net = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(size, n_classes))

    def forward(self, x):
        return self.net(x)


class E1(nn.Module):
    """One GRU layer over the P1 features of the frames of a window, then the output layer (D29)."""

    def __init__(self, n_classes=4, feature_size=128, hidden=64):
        super().__init__()
        self.gru = nn.GRU(feature_size, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_classes)

    def forward(self, x):  # x: windows x frames x feature size
        return self.out(self.gru(x)[1][-1])  # the state after the last frame


MODELS = {"p1": P1, "b1": B1, "e1": E1}


def build(cfg, **override):
    """Create the model that a config names. Each model gets only the config values that it knows."""
    model = MODELS[cfg["model"]]
    known = inspect.signature(model.__init__).parameters
    return model(**{k: v for k, v in (cfg | override).items() if k in known})
