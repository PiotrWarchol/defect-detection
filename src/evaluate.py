import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import os
from src.data_prep import get_dataloaders
from src.model import build_model, get_device

def load_best_model(device):
    """Load the best saved model from training."""
    model = build_model(num_classes=2, freeze_backbone=False)
    model.load_state_dict(torch.load('models/best_model.pth', 
                                      map_location=device))
    model = model.to(device)
    model.eval()
    print("Best model loaded successfully!")
    return model

def get_predictions(model, loader, device):
    """Run inference on entire dataset and collect predictions."""
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            
            # Get probabilities using softmax
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return (np.array(all_preds), 
            np.array(all_labels), 
            np.array(all_probs))

def plot_confusion_matrix(labels, preds, classes):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, preds)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=12)
    ax.set_yticklabels(classes, fontsize=12)

    # Add text annotations
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]),
                   ha='center', va='center',
                   fontsize=16, fontweight='bold', color=color)

    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('True Label', fontsize=13)
    ax.set_title('Manufacturing Defect Detection\nConfusion Matrix', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/confusion_matrix.png', dpi=150, 
                bbox_inches='tight')
    plt.show()
    print("Confusion matrix saved to outputs/confusion_matrix.png")
    
    return cm

def plot_training_history():
    """Plot training and test accuracy/loss curves."""
    import json
    
    with open('models/training_history.json', 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_acc']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    ax1.plot(epochs, history['train_acc'], 'b-o', 
             label='Train Accuracy', linewidth=2)
    ax1.plot(epochs, history['test_acc'], 'r-o', 
             label='Test Accuracy', linewidth=2)
    ax1.axvline(x=5.5, color='gray', linestyle='--', 
                label='Phase 1 → Phase 2', alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Training and Test Accuracy', fontsize=13, 
                  fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([60, 102])

    # Loss plot
    ax2.plot(epochs, history['train_loss'], 'b-o', 
             label='Train Loss', linewidth=2)
    ax2.plot(epochs, history['test_loss'], 'r-o', 
             label='Test Loss', linewidth=2)
    ax2.axvline(x=5.5, color='gray', linestyle='--', 
                label='Phase 1 → Phase 2', alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training and Test Loss', fontsize=13, 
                  fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Manufacturing Defect Detection — Training History', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('outputs/training_history.png', dpi=150, 
                bbox_inches='tight')
    plt.show()
    print("Training history saved to outputs/training_history.png")

def plot_sample_predictions(model, loader, device, classes, n_samples=12):
    """Plot sample predictions with confidence scores."""
    import torchvision

    model.eval()
    images_shown = 0
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            for i in range(len(images)):
                if images_shown >= n_samples:
                    break

                # Denormalize image for display
                img = images[i].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406])
                std = torch.tensor([0.229, 0.224, 0.225])
                img = img * std[:, None, None] + mean[:, None, None]
                img = img.permute(1, 2, 0).numpy()
                img = np.clip(img, 0, 1)

                pred = preds[i].item()
                label = labels[i].item()
                confidence = probs[i][pred].item() * 100
                correct = pred == label

                axes[images_shown].imshow(img)
                axes[images_shown].axis('off')

                color = 'green' if correct else 'red'
                status = '✓' if correct else '✗'
                title = (f"{status} Pred: {classes[pred]}\n"
                        f"True: {classes[label]}\n"
                        f"Conf: {confidence:.1f}%")
                axes[images_shown].set_title(
                    title, fontsize=9, color=color, fontweight='bold'
                )
                images_shown += 1

            if images_shown >= n_samples:
                break

    plt.suptitle('Sample Predictions — Manufacturing Defect Detection',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/sample_predictions.png', dpi=150, 
                bbox_inches='tight')
    plt.show()
    print("Sample predictions saved to outputs/sample_predictions.png")

def evaluate_model():
    """Run complete model evaluation."""
    print("Running model evaluation...")
    print("=" * 50)

    device = get_device()
    _, test_loader, classes = get_dataloaders()
    model = load_best_model(device)

    print("\nRunning inference on test set...")
    preds, labels, probs = get_predictions(model, test_loader, device)

    # Core metrics
    accuracy = accuracy_score(labels, preds) * 100
    precision = precision_score(labels, preds, 
                               average='weighted') * 100
    recall = recall_score(labels, preds, 
                         average='weighted') * 100
    f1 = f1_score(labels, preds, average='weighted') * 100

    # Per class metrics - critical for manufacturing
    precision_per_class = precision_score(
        labels, preds, average=None
    ) * 100
    recall_per_class = recall_score(
        labels, preds, average=None
    ) * 100

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Overall Accuracy:  {accuracy:.2f}%")
    print(f"Weighted Precision: {precision:.2f}%")
    print(f"Weighted Recall:    {recall:.2f}%")
    print(f"Weighted F1 Score:  {f1:.2f}%")
    print()
    print("Per-Class Performance:")
    for i, class_name in enumerate(classes):
        print(f"  {class_name}:")
        print(f"    Precision: {precision_per_class[i]:.2f}%")
        print(f"    Recall:    {recall_per_class[i]:.2f}%")
    print()
    print("Full Classification Report:")
    print(classification_report(labels, preds, 
                                target_names=classes))

    # Manufacturing context note
    defect_idx = classes.index('def_front')
    defect_recall = recall_per_class[defect_idx]
    print("=" * 50)
    print("MANUFACTURING CONTEXT")
    print("=" * 50)
    print(f"Defect Recall: {defect_recall:.2f}%")
    if defect_recall >= 99:
        print("EXCELLENT — Missing less than 1% of defective parts")
    elif defect_recall >= 95:
        print("GOOD — Missing less than 5% of defective parts")
    else:
        print("NEEDS IMPROVEMENT — Too many defects being missed")
    print()

    # Generate visualizations
    print("Generating visualizations...")
    cm = plot_confusion_matrix(labels, preds, classes)
    plot_training_history()
    plot_sample_predictions(model, test_loader, device, classes)

    print("\nEvaluation complete!")
    print("All outputs saved to outputs/ folder")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

if __name__ == "__main__":
    evaluate_model()