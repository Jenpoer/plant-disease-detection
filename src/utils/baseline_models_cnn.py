"""
This module provides a factory function for loading and customizing CNN baseline models.

Supported architectures:
- MobileNetV3 Small: Optimized for speed and low-resource devices.
- EfficientNet-B0: Balanced performance focused on accuracy and efficiency.

The script replaces the final classification layer to match the
required number of classes for the Plant Disease Detection project (default: 26).
"""

import torch.nn as nn
from torchvision import models


def get_model(model_name: str, num_classes: int = 26, pretrained: bool = True):
    """
    Entry point to get a CNN model.
    Supported: 'mobilenet_v3_small', 'efficientnet_b0'
    """
    weights = "DEFAULT" if pretrained else None

    if model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=weights)
        # MobileNetV3 classifier:
        # (classifier): Sequential(
        #   (0): Linear(...)
        #   (1): Hardswish()
        #   (2): Dropout(...)
        #   (3): Linear(in_features=1024, out_features=1000, bias=True)
        # )
        # HEAD REPLACEMENT
        # Original classifier: [Linear -> Hardswish -> Dropout -> Linear(1024, 1000)]
        # We replace the last layer (index 3) to output 'num_classes' instead of 1000.
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        # EfficientNet classifier:
        # (classifier): Sequential(
        #   (0): Dropout(...)
        #   (1): Linear(in_features=1280, out_features=1000, bias=True)
        # )
        # HEAD REPLACEMENT
        # Original classifier: [Dropout -> Linear(1280, 1000)]
        # We replace the last layer (index 1) to output 'num_classes' instead of 1000.
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model
