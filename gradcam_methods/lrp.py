import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import cv2

class LRP:
    """
    Layerwise Relevance Propagation implementation for ResNet models
    Based on the paper: "Layer-Wise Relevance Propagation for Neural Networks with Local Renormalization Layers"
    """
    def __init__(self, model):
        self.model = model
        self.model.eval()
        
        # Save intermediate activations
        self.activations = {}
        self.register_hooks()
    
    def register_hooks(self):
        """Register forward hooks to capture activations"""
        def hook_fn(name):
            def hook(module, input, output):
                self.activations[name] = output
            return hook
        
        # Register hooks for all layers we need for LRP
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                module.register_forward_hook(hook_fn(name))
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate LRP heatmap for the input image
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Class index to generate LRP for. If None, uses the max prediction
            
        Returns:
            heatmap: LRP heatmap (H, W)
        """
        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Get target class if not specified
        if target_class is None:
            target_class = outputs.argmax(dim=1).item()
        
        # For a simplified approach, we'll use the gradient*input method
        # This is a simpler version of LRP that still provides meaningful relevance scores
        input_tensor.requires_grad = True
        outputs = self.model(input_tensor)
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(outputs)
        one_hot[0, target_class] = 1
        
        # Backward pass
        outputs.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and multiply by input (similar to Gradient * Input rule)
        gradients = input_tensor.grad.detach()
        saliency = torch.abs(gradients * input_tensor)
        
        # Sum across channels for visualization
        heatmap = torch.sum(saliency, dim=1).squeeze().cpu().numpy()
        
        # Normalize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap


def visualize_lrp(img_tensor, lrp_heatmap, class_name="Predicted Class"):
    """
    Visualize LRP results
    
    Args:
        img_tensor: Input image tensor
        lrp_heatmap: LRP heatmap
        class_name: Name of class for display
    """
    # Convert tensor to image
    img = img_tensor[0].permute(1, 2, 0).cpu().numpy()
    img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    img = img.astype(np.uint8)
    
    # Create heatmap overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * lrp_heatmap), cv2.COLORMAP_HOT)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on original image
    alpha = 0.5
    overlay = img * (1 - alpha) + heatmap * alpha
    overlay = overlay.astype(np.uint8)
    
    # Display results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(lrp_heatmap, cmap='hot')
    plt.title(f"LRP Heatmap: {class_name}")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def example_lrp(model, img_path, class_idx_pembroke=0, class_idx_cardigan=1, device="cpu"):
    """Example of using LRP on a test image"""
    # Preprocessing function
    from torchvision import transforms
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    # Generate prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        pembroke_prob = probs[class_idx_pembroke].item()
        cardigan_prob = probs[class_idx_cardigan].item()
        predicted_class = outputs.argmax(dim=1).item()
        class_name = "Pembroke" if predicted_class == class_idx_pembroke else "Cardigan"
    
    # Initialize LRP
    lrp = LRP(model)
    
    # Generate LRP heatmap for predicted class
    heatmap = lrp.generate(img_tensor, predicted_class)
    
    # Visualize results
    print(f"Prediction: Pembroke: {pembroke_prob:.2%}, Cardigan: {cardigan_prob:.2%}")
    visualize_lrp(img_tensor, heatmap, class_name=class_name)
    
    return heatmap


def compare_methods(model, img_path, class_idx_pembroke=0, class_idx_cardigan=1, device="cpu"):
    """
    Compare GradCAM and LRP visualizations for the same image
    
    Args:
        model: PyTorch model
        img_path: Path to input image
        class_idx_pembroke: Class index for Pembroke
        class_idx_cardigan: Class index for Cardigan
        device: Device to run on
    """
    # Import GradCAM
    from gradcam_methods.gradcam import GradCAM, generate_contrastive_gradcam
    
    # Preprocessing
    from torchvision import transforms
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        pembroke_prob = probs[class_idx_pembroke].item()
        cardigan_prob = probs[class_idx_cardigan].item()
        predicted_class = outputs.argmax(dim=1).item()
        class_name = "Pembroke" if predicted_class == class_idx_pembroke else "Cardigan"
        
    print(f"Prediction: {class_name} (Pembroke: {pembroke_prob:.2%}, Cardigan: {cardigan_prob:.2%})")
    
    # Generate GradCAM visualization
    grad_cam = GradCAM(model)
    gradcam_heatmap = grad_cam.generate(img_tensor, predicted_class)
    
    # Generate LRP visualization
    lrp = LRP(model)
    lrp_heatmap = lrp.generate(img_tensor, predicted_class)
    
    # Convert tensor to image for display
    img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    img_np = img_np.astype(np.uint8)
    
    # Create heatmap overlays
    gradcam_overlay = cv2.applyColorMap(np.uint8(255 * gradcam_heatmap), cv2.COLORMAP_JET)
    gradcam_overlay = cv2.cvtColor(gradcam_overlay, cv2.COLOR_BGR2RGB)
    
    lrp_overlay = cv2.applyColorMap(np.uint8(255 * lrp_heatmap), cv2.COLORMAP_HOT)
    lrp_overlay = cv2.cvtColor(lrp_overlay, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmaps on original image
    alpha = 0.5
    gradcam_result = img_np * (1 - alpha) + gradcam_overlay * alpha
    lrp_result = img_np * (1 - alpha) + lrp_overlay * alpha
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img_np)
    plt.title(f"Original - Predicted: {class_name}")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(gradcam_heatmap, cmap='jet')
    plt.title("Grad-CAM Heatmap")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(lrp_heatmap, cmap='hot')
    plt.title("LRP Heatmap")
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(gradcam_result.astype(np.uint8))
    plt.title("Grad-CAM Overlay")
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(lrp_result.astype(np.uint8))
    plt.title("LRP Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Return all heatmaps for further analysis
    return {
        "gradcam": gradcam_heatmap,
        "lrp": lrp_heatmap
    } 