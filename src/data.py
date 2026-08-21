from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: list[str]


def train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _make_weighted_sampler(targets: np.ndarray) -> WeightedRandomSampler:
    class_counts = np.bincount(targets)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[targets]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(
    data_dir: str,
    image_size: int = 224,
    batch_size: int = 32,
    val_size: float = 0.2,
    seed: int = 42,
    num_workers: int = 2,
    balanced_sampling: bool = False,
) -> DataLoaders:
    root = Path(data_dir)
    train_dir = root / "TRAIN"
    test_dir = root / "TEST"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"Expected TRAIN and TEST folders under {root}. "
            "Pass the directory that directly contains TRAIN/ and TEST/."
        )

    # Create the split on ORIGINAL images before any resampling or duplication.
    base = datasets.ImageFolder(train_dir)
    targets = np.asarray(base.targets)
    indices = np.arange(len(base))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        stratify=targets,
        random_state=seed,
    )

    train_full = datasets.ImageFolder(train_dir, transform=train_transform(image_size))
    val_full = datasets.ImageFolder(train_dir, transform=eval_transform(image_size))
    test_set = datasets.ImageFolder(test_dir, transform=eval_transform(image_size))

    train_set = Subset(train_full, train_idx.tolist())
    val_set = Subset(val_full, val_idx.tolist())

    sampler = None
    shuffle = True
    if balanced_sampling:
        sampler = _make_weighted_sampler(targets[train_idx])
        shuffle = False

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if train_full.classes != test_set.classes:
        raise ValueError("TRAIN and TEST class folders do not match.")

    return DataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        class_names=train_full.classes,
    )
