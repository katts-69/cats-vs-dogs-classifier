"""Building the EfficientNet-B0 model and loading trained weights."""
import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

# Must match the class folder names, which ImageFolder sorts alphabetically
CLASS_NAMES = ["Cat", "Dog"]


def build_model(pretrained=True):
    """EfficientNet-B0 with its 1,000-class ImageNet output swapped for a Cat/Dog output.

    pretrained=True starts from weights trained on ImageNet (for training).
    pretrained=False skips that download (for loading our own trained weights).
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
    return model


def set_backbone_trainable(model, trainable):
    """Freeze (False) or unfreeze (True) the pretrained layers. The new final layer always trains."""
    for param in model.features.parameters():
        param.requires_grad = trainable


def load_model(path, device="cpu"):
    """Load trained weights from `path`, ready to make predictions."""
    model = build_model(pretrained=False)
    # weights_only=True loads plain numbers only. A .pt file can otherwise contain code that
    # runs when it is loaded, which matters because this file is downloaded from the internet.
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()
