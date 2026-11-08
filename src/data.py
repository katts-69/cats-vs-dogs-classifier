"""Loading the dataset and preparing images for the model."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGE_SIZE = 224
SPLITS = ("train", "validation", "test")

# Pretrained torchvision models expect images normalized with these ImageNet statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Training images get random changes (augmentation) so the model learns that a cat is
# still a cat when zoomed in or mirrored
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Validation, test and real-world images are always prepared the same way, with no randomness
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_datasets(data_dir, splits=SPLITS):
    """Load each split from data_dir/<split>/<class name>/*.jpg.

    ImageFolder takes each label from the folder name and loads every image as RGB.
    """
    data_dir = Path(data_dir)
    return {
        split: datasets.ImageFolder(
            data_dir / split,
            transform=train_transform if split == "train" else eval_transform,
        )
        for split in splits
    }


def make_loader(dataset, batch_size=32, shuffle=False, num_workers=4, seed=42):
    """Wrap a dataset in a DataLoader that serves it in batches."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )
