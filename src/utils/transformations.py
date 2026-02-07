"""
This module provides data transformation pipelines for different model architectures.
TODO for M4: Create builder design to build a transform pipeline based on configuration.
"""

import torch
from torchvision import transforms
from timm.data import create_transform

def get_default_transforms(model_name: str, image_size: int = 224):
    """
    Returns appropriate default data transformations for the given model.

    Args:
        model_name: Name of the model (e.g., 'mobilenet_v3_small', 'efficientnet_b0', 'vit_base_patch16_224')
        image_size: Target image size for resizing (default: 224)   
    Returns:
        A tuple of (train_transform, val_transform) suitable for the specified model.
    """

    if model_name in ["mobilenet_v3_small", "efficientnet_b0"]:
        # Transformations:
        #- Training data: Includes random horizontal flips for augmentation
        #- Validation data: Deterministic transforms only (no augmentation)
        #- Both use ImageNet normalization to match pre-trained model expectations

        # Standard ImageNet normalization statistics for RGB channels
        # These specific mean/std values are required because the pre-trained models
        # (MobileNet, EfficientNet) were trained on ImageNet using this distribution.
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )


    elif model_name in ["vit_base_patch16_224"]:
        # Use timm's create_transform for ViT
        train_transform = create_transform(
            input_size=image_size,
            is_training=True
        )

        val_transform = create_transform(
            input_size=image_size,
            is_training=False
        )

        test_transform = create_transform(
            input_size=image_size,
            is_training=False
        )

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return train_transform, val_transform, test_transform