"""
This module provides data transformation pipelines for different model architectures.
TODO for M4: Create builder design to build a transform pipeline based on configuration.
"""

import torch
from torchvision import transforms
from timm.data import create_transform

def get_transforms(model_name: str, image_size: int = 224, transforms_config: dict = None):
    """
    Returns appropriate data transformations for the given model.

    Args:
        model_name: Name of the model (e.g., 'mobilenet_v3_small', 'efficientnet_b0', 'vit_base_patch16_224', 'cct_14_7x2_224')
        image_size: Target image size for resizing (default: 224)
        transforms_config: Optional dictionary to customize transformations
    Returns:
        A tuple of (train_transform, val_transform, test_transform) suitable for the specified model.
    """

    transform_steps = []

    if transforms_config:
        # If transforms_config is provided, apply custom transformations
        for step in transforms_config.get("steps", []):
            if step["name"] == "color_jitter":
                    brightness = step["params"].get("brightness", 0)
                    contrast = step["params"].get("contrast", 0)
                    saturation = step["params"].get("saturation", 0)
                    hue = step["params"].get("hue", 0)
                    transform_steps.append(transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue))

            elif step["name"] == "random_rotation":
                degrees = step["params"].get("degrees", 0)
                transform_steps.append(transforms.RandomRotation(degrees))

            # Add more custom transformations as needed

    train_transform, val_transform, test_transform = get_default_transforms(model_name, image_size)

    if transform_steps:
        # Apply custom transforms to each transform pipeline
        train_transform = transforms.Compose(transform_steps + [train_transform.transforms[-1]])  # Keep normalization at the end
        val_transform = transforms.Compose(transform_steps + [val_transform.transforms[-1]])
        test_transform = transforms.Compose(transform_steps + [test_transform.transforms[-1]])

    return train_transform, val_transform, test_transform

def get_default_transforms(model_name: str, image_size: int = 224):
    """
    Returns appropriate default data transformations for the given model.

    Args:
        model_name: Name of the model (e.g., 'mobilenet_v3_small', 'efficientnet_b0', 'vit_base_patch16_224', 'cct_14_7x2_224')
        image_size: Target image size for resizing (default: 224)   
    Returns:
        A tuple of (train_transform, val_transform) suitable for the specified model.
    """

    if model_name in ["mobilenet_v3_small", "efficientnet_b0", "cct_14_7x2_224"]:
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