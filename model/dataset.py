"""
Step 3: Dataset & DataLoader
- Custom PyTorch Dataset for real/fake face images
- Augmentations for training
- ImageNet normalization
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ── Transforms ────────────────────────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_train_transforms(image_size: int = 224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.05),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        T.RandomRotation(degrees=5),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_val_transforms(image_size: int = 224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def denormalize(tensor):
    """Reverse ImageNet normalization for visualization."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


# ── Dataset ───────────────────────────────────────────────────────────────────

class DeepfakeDataset(Dataset):
    """
    Expects folder structure:
        root/
          real/  *.jpg
          fake/  *.jpg
    """
    LABEL_MAP = {"real": 0, "fake": 1}

    def __init__(self, root: str, transform=None):
        self.root      = Path(root)
        self.transform = transform
        self.samples   = []   # list of (path, label_int)

        for label_name, label_int in self.LABEL_MAP.items():
            label_dir = self.root / label_name
            if not label_dir.exists():
                raise FileNotFoundError(f"Directory not found: {label_dir}")
            for img_path in sorted(label_dir.glob("*.jpg")):
                self.samples.append((img_path, label_int))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {self.root}")

        # class counts for weighted sampler
        self.class_counts = [0, 0]
        for _, lbl in self.samples:
            self.class_counts[lbl] += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_sample_weights(self):
        """For WeightedRandomSampler to handle class imbalance."""
        total = len(self.samples)
        weights_per_class = [total / c for c in self.class_counts]
        sample_weights = [weights_per_class[lbl] for _, lbl in self.samples]
        return torch.tensor(sample_weights, dtype=torch.float)


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(data_root: str, batch_size: int = 32, num_workers: int = 4,
                    image_size: int = 224, use_weighted_sampler: bool = True):
    """
    Returns dict of DataLoaders for train / val / test splits.
    data_root should contain train/ val/ test/ subdirectories.
    """
    data_root = Path(data_root)

    train_dataset = DeepfakeDataset(data_root / "train", transform=get_train_transforms(image_size))
    val_dataset   = DeepfakeDataset(data_root / "val",   transform=get_val_transforms(image_size))
    test_dataset  = DeepfakeDataset(data_root / "test",  transform=get_val_transforms(image_size))

    # Weighted sampler for imbalanced classes
    if use_weighted_sampler:
        from torch.utils.data import WeightedRandomSampler
        weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_shuffle  = False
        train_sampler  = sampler
    else:
        train_shuffle  = True
        train_sampler  = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset):,} images | Val: {len(val_dataset):,} | Test: {len(test_dataset):,}")
    print(f"  Train class counts → real: {train_dataset.class_counts[0]:,} | fake: {train_dataset.class_counts[1]:,}")

    return {"train": train_loader, "val": val_loader, "test": test_loader}


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "data/processed"
    loaders = get_dataloaders(root, batch_size=8, num_workers=0)
    imgs, labels = next(iter(loaders["train"]))
    print(f"Batch shape: {imgs.shape}  Labels: {labels.tolist()}")
