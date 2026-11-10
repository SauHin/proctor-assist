"""Models. Every model returns class scores (logits) for the 4 classes of D14."""
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


MODELS = {"p1": P1}
