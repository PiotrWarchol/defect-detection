import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes=2, freeze_backbone=True):
    """
    Build a ResNet18 model fine-tuned for defect detection.
    
    We use transfer learning — start with ResNet18 pretrained on 
    ImageNet and replace the final layer for our 2-class problem.
    
    Args:
        num_classes: Number of output classes (2 = defective/normal)
        freeze_backbone: If True, only train the final layer initially
    
    Returns:
        model: PyTorch model ready for training
    """
    # Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    if freeze_backbone:
        # Freeze all layers except the final classifier
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the final fully connected layer
    # ResNet18's final layer outputs 512 features
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )
    
    return model

def unfreeze_model(model):
    """
    Unfreeze all model layers for fine-tuning.
    Called after initial training to allow full model updates.
    """
    for param in model.parameters():
        param.requires_grad = True
    return model

def count_parameters(model):
    """Count trainable parameters in the model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() 
                   if p.requires_grad)
    return total, trainable

def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU — training will be slower but will work fine")
    return device

if __name__ == "__main__":
    print("Building defect detection model...")
    print("=" * 40)
    
    device = get_device()
    model = build_model(num_classes=2, freeze_backbone=True)
    model = model.to(device)
    
    total, trainable = count_parameters(model)
    print(f"\nModel: ResNet18 with custom classifier")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Frozen parameters:    {total - trainable:,}")
    print()
    
    # Test forward pass with dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output classes: defective={output[0][0]:.4f}, normal={output[0][1]:.4f}")
    print()
    print("Model built successfully!")