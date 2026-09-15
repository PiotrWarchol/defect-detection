import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from torchvision import transforms
import os
import cv2

from src.model import build_model, get_device

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    Generates heatmaps showing which regions of an image
    the model focused on when making its prediction.
    
    In manufacturing context: highlights exactly WHERE
    on the casting the defect was detected.
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to capture gradients and activations
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Save forward pass activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Save backward pass gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate(self, image_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap for an image.
        
        Args:
            image_tensor: Preprocessed image tensor (1, 3, 224, 224)
            class_idx: Target class (None = use predicted class)
        
        Returns:
            heatmap: Normalized heatmap array (224, 224)
            pred_class: Predicted class index
            confidence: Prediction confidence
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(image_tensor)
        probs = torch.softmax(output, dim=1)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        confidence = probs[0][class_idx].item()
        
        # Backward pass for target class
        self.model.zero_grad()
        output[0][class_idx].backward()
        
        # Generate heatmap
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        
        # Weighted combination of activation maps
        heatmap = (weights * self.activations).sum(dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
        
        # Normalize to 0-1
        heatmap = heatmap.squeeze().cpu().numpy()
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (
                heatmap.max() - heatmap.min()
            )
        
        # Resize to image size
        heatmap = cv2.resize(heatmap, (224, 224))
        
        return heatmap, class_idx, confidence


def get_gradcam_model(device):
    """Build model and attach Grad-CAM to the last conv layer."""
    model = build_model(num_classes=2, freeze_backbone=False)
    model.load_state_dict(
        torch.load('models/best_model.pth', map_location=device)
    )
    model = model.to(device)
    
    # Target the last convolutional layer of ResNet18
    # layer4[-1].conv2 is the final feature extraction layer
    target_layer = model.layer4[-1].conv2
    
    gradcam = GradCAM(model, target_layer)
    return model, gradcam


def preprocess_image(image_path):
    """Load and preprocess an image for inference."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    return image, tensor


def overlay_heatmap(original_image, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image.
    
    Red areas = high attention (where defect detected)
    Blue areas = low attention
    """
    # Convert PIL image to numpy
    img_array = np.array(original_image.resize((224, 224)))
    
    # Apply colormap to heatmap
    colormap = plt.get_cmap('jet')
    heatmap_colored = colormap(heatmap)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
    
    # Blend heatmap with original image
    overlay = (alpha * heatmap_colored + 
               (1 - alpha) * img_array).astype(np.uint8)
    
    return overlay


def visualize_gradcam_batch(n_defective=4, n_normal=4):
    """
    Generate Grad-CAM visualizations for sample images.
    Shows both defective and normal castings with heatmaps.
    """
    device = get_device()
    model, gradcam = get_gradcam_model(device)
    
    classes = ['def_front', 'ok_front']
    base_path = 'data/casting_data/casting_data/test'
    
    # Collect sample images
    samples = []
    
    # Get defective samples
    def_path = f'{base_path}/def_front'
    def_images = os.listdir(def_path)[:n_defective]
    for img_name in def_images:
        samples.append({
            'path': f'{def_path}/{img_name}',
            'true_label': 0,
            'true_name': 'def_front'
        })
    
    # Get normal samples
    ok_path = f'{base_path}/ok_front'
    ok_images = os.listdir(ok_path)[:n_normal]
    for img_name in ok_images:
        samples.append({
            'path': f'{ok_path}/{img_name}',
            'true_label': 1,
            'true_name': 'ok_front'
        })
    
    # Generate visualizations
    n_samples = len(samples)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, n_samples * 4))
    
    col_titles = ['Original Image', 'Grad-CAM Heatmap', 'Overlay']
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    
    for idx, sample in enumerate(samples):
        # Load and preprocess
        original_image, tensor = preprocess_image(sample['path'])
        tensor = tensor.to(device)
        
        # Generate Grad-CAM
        heatmap, pred_class, confidence = gradcam.generate(tensor)
        
        # Create overlay
        overlay = overlay_heatmap(original_image, heatmap)
        
        # Determine if prediction is correct
        correct = pred_class == sample['true_label']
        status = '✓ Correct' if correct else '✗ Wrong'
        color = 'green' if correct else 'red'
        
        # Row label
        row_label = (f"True: {sample['true_name']}\n"
                    f"Pred: {classes[pred_class]}\n"
                    f"Conf: {confidence*100:.1f}%\n"
                    f"{status}")
        
        # Plot original
        axes[idx, 0].imshow(original_image.resize((224, 224)))
        axes[idx, 0].set_ylabel(row_label, fontsize=9, 
                                color=color, fontweight='bold',
                                rotation=0, labelpad=120,
                                va='center')
        axes[idx, 0].axis('off')
        
        # Plot heatmap
        axes[idx, 1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
        axes[idx, 1].axis('off')
        
        # Plot overlay
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].axis('off')
    
    plt.suptitle(
        'Grad-CAM Visualization — Manufacturing Defect Detection\n'
        'Red regions indicate areas the model focused on for classification',
        fontsize=13, fontweight='bold', y=1.01
    )
    
    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/gradcam_visualization.png', 
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Grad-CAM visualization saved to outputs/gradcam_visualization.png")


def analyze_single_image(image_path):
    """
    Analyze a single image and show detailed Grad-CAM output.
    Use this to test the model on any casting image.
    """
    device = get_device()
    model, gradcam = get_gradcam_model(device)
    classes = ['DEFECTIVE', 'NORMAL']
    
    print(f"Analyzing: {image_path}")
    
    # Load and process
    original_image, tensor = preprocess_image(image_path)
    tensor = tensor.to(device)
    
    # Generate Grad-CAM
    heatmap, pred_class, confidence = gradcam.generate(tensor)
    overlay = overlay_heatmap(original_image, heatmap)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_image.resize((224, 224)))
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    result = classes[pred_class]
    color = 'red' if pred_class == 0 else 'green'
    
    plt.suptitle(
        f'Prediction: {result} | Confidence: {confidence*100:.1f}%',
        fontsize=14, fontweight='bold', color=color
    )
    
    plt.tight_layout()
    plt.savefig('outputs/single_analysis.png', dpi=150, 
                bbox_inches='tight')
    plt.show()
    
    print(f"\nResult: {result}")
    print(f"Confidence: {confidence*100:.1f}%")
    print("Saved to outputs/single_analysis.png")
    
    return result, confidence


if __name__ == "__main__":
    print("Generating Grad-CAM visualizations...")
    print("=" * 50)
    visualize_gradcam_batch(n_defective=4, n_normal=4)
    print("\nGrad-CAM generation complete!")