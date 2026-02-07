import torch
from torchvision import transforms
from timm.data import create_transform

def get_transforms(model_name: str, image_size: int = 224, **kwargs):
    """
    Returns appropriate data transformations for the given model.

    Args:
        model_name: Name of the model (e.g., 'mobilenet_v3_small', 'efficientnet_b0', 'vit_base_patch16_224')
        image_size: Target image size for resizing (default: 224)   
    Returns:
        A tuple of (train_transform, val_transform) suitable for the specified model.
    """

    if model_name in ["mobilenet_v3_small", "efficientnet_b0"]:
        # Standard transforms for CNNs
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

    elif model_name == "vit_base_patch16_224":
        # Use timm's create_transform for ViT
        train_transform = create_transform(
            input_size=image_size,
            is_training=True
        )

        val_transform = create_transform(
            input_size=image_size,
            is_training=False
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return train_transform, val_transform