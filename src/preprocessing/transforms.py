
"""
Define image transformations for training, validation, and test datasets.

This module contains two torchvision transformation pipelines:

- train_transform:
    Applies resizing, data augmentation, and tensor conversion to training
    images. The augmentation operations include random rotations, horizontal
    flips, affine transformations, and small brightness/contrast variations.

- transform:
    Applies only resizing and tensor conversion. This pipeline is intended
    for validation and test images, where random data augmentation should not
    be used.

The images are resized to 224x224 pixels and converted to PyTorch tensors
with pixel values scaled from [0, 255] to [0, 1].
"""

from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    transforms.ToTensor(),
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])