import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import os
from src.data_prep import get_dataloaders
from src.model import build_model, unfreeze_model, get_device, count_parameters

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}/{len(loader)} "
                  f"| Loss: {total_loss/(batch_idx+1):.4f} "
                  f"| Acc: {100.*correct/total:.1f}%")

    return total_loss / len(loader), 100. * correct / total

def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataset and return loss and accuracy."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), 100. * correct / total

def train(epochs_frozen=5, epochs_unfrozen=5):
    """
    Full training pipeline with two phases:
    Phase 1: Train only the classifier head (backbone frozen)
    Phase 2: Fine-tune the entire network (all layers unfrozen)
    """
    print("Starting defect detection model training...")
    print("=" * 50)

    # Setup
    device = get_device()
    train_loader, test_loader, classes = get_dataloaders()
    print(f"Classes: {classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Build model
    model = build_model(num_classes=2, freeze_backbone=True)
    model = model.to(device)
    total, trainable = count_parameters(model)
    print(f"\nTrainable parameters: {trainable:,} of {total:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    # Track best model and history
    best_accuracy = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }

    # ─────────────────────────────────────────
    # Phase 1 — Train classifier head only
    # ─────────────────────────────────────────
    print(f"\nPhase 1 — Training classifier head ({epochs_frozen} epochs)")
    print("-" * 50)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.5
    )

    for epoch in range(epochs_frozen):
        start = time.time()
        print(f"\nEpoch {epoch + 1}/{epochs_frozen}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - start
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.1f}%")
        print(f"  Time: {elapsed:.0f}s")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f"  New best model saved! Accuracy: {best_accuracy:.1f}%")

    # ─────────────────────────────────────────
    # Phase 2 — Fine-tune entire network
    # ─────────────────────────────────────────
    print(f"\nPhase 2 — Fine-tuning entire network ({epochs_unfrozen} epochs)")
    print("-" * 50)

    model = unfreeze_model(model)
    total, trainable = count_parameters(model)
    print(f"Trainable parameters: {trainable:,} of {total:,}")

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.5
    )

    for epoch in range(epochs_unfrozen):
        start = time.time()
        print(f"\nEpoch {epoch + 1}/{epochs_unfrozen}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()

        elapsed = time.time() - start
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.1f}%")
        print(f"  Time: {elapsed:.0f}s")

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
            print(f"  New best model saved! Accuracy: {best_accuracy:.1f}%")

    # Save training history
    with open('models/training_history.json', 'w') as f:
        json.dump(history, f)

    print("\n" + "=" * 50)
    print(f"Training complete!")
    print(f"Best test accuracy: {best_accuracy:.1f}%")
    print(f"Model saved to: models/best_model.pth")

    return model, history

if __name__ == "__main__":
    model, history = train(epochs_frozen=5, epochs_unfrozen=5)