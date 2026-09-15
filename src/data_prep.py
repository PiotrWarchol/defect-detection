import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Dataset paths
DATA_DIR = 'data/casting_data/casting_data'
TRAIN_DIR = f'{DATA_DIR}/train'
TEST_DIR = f'{DATA_DIR}/test'

# Image settings
IMAGE_SIZE = 224  # ResNet expects 224x224
BATCH_SIZE = 32

def get_transforms():
    """
    Define image transformations for training and testing.
    Training includes augmentation to improve generalization.
    Testing uses only normalization — no augmentation.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transforms, test_transforms

def get_dataloaders():
    """
    Create PyTorch DataLoaders for train and test sets.
    Returns dataloaders and class names.
    """
    train_transforms, test_transforms = get_transforms()

    # Load datasets using ImageFolder
    # ImageFolder automatically assigns labels based on folder names
    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transforms
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=test_transforms
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader, train_dataset.classes

def verify_data():
    """
    Verify dataset loaded correctly and print summary statistics.
    """
    train_loader, test_loader, classes = get_dataloaders()

    print("Dataset Summary")
    print("=" * 40)
    print(f"Classes: {classes}")
    print(f"Class to index mapping: {train_loader.dataset.class_to_idx}")
    print()
    print(f"Training samples:  {len(train_loader.dataset)}")
    print(f"Test samples:      {len(test_loader.dataset)}")
    print(f"Training batches:  {len(train_loader)}")
    print(f"Test batches:      {len(test_loader)}")
    print()

    # Check class distribution
    train_counts = {}
    for _, label in train_loader.dataset.samples:
        class_name = classes[label]
        train_counts[class_name] = train_counts.get(class_name, 0) + 1

    print("Training class distribution:")
    for class_name, count in train_counts.items():
        pct = count / len(train_loader.dataset) * 100
        print(f"  {class_name}: {count} images ({pct:.1f}%)")
    print()

    # Load one batch and verify shapes
    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")
    print(f"Image pixel range: [{images.min():.2f}, {images.max():.2f}]")
    print()
    print("Data preparation verified successfully!")

    return train_loader, test_loader, classes

if __name__ == "__main__":
    verify_data()