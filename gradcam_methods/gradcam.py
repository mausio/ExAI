import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import cv2

class GradCAM:
    """
    Grad-CAM implementation for ResNet model visualization
    Based on the paper: Grad-CAM: Visual Explanations from Deep Networks
    """
    def __init__(self, model, target_layer_name="layer4"):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        
        # Save gradients and activations
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks to the target layer"""
        
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Get the target layer
        target_layer = None
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                target_layer = module
                break
        
        if target_layer is None:
            raise ValueError(f"Layer {self.target_layer_name} not found in model!")
            
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(forward_hook)
        self.backward_hook = target_layer.register_backward_hook(backward_hook)
    
    def _release_hooks(self):
        """Remove hooks to prevent memory leaks"""
        self.forward_hook.remove()
        self.backward_hook.remove()
    
    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap for the input image
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Class index to generate CAM for. If None, uses the max prediction
            
        Returns:
            heatmap: Grad-CAM heatmap (H, W)
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients.detach().cpu().numpy()[0]  # (C, H, W)
        activations = self.activations.detach().cpu().numpy()[0]  # (C, H, W)
        
        # Weight the channels by average gradient
        weights = np.mean(gradients, axis=(1, 2))  # (C,)
        
        # Weight activations by weights
        cam = np.zeros(activations.shape[1:], dtype=np.float32)  # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU to highlight positive contributions
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize to input size
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        
        return cam
    
    def __del__(self):
        """Clean up when object is deleted"""
        try:
            self._release_hooks()
        except:
            pass


def generate_contrastive_gradcam(model, img_tensor, class_idx_a, class_idx_b, layer_name="layer4"):
    """
    Generate contrastive Grad-CAM between two classes
    
    Args:
        model: PyTorch model
        img_tensor: Input image tensor
        class_idx_a: First class index (Pembroke)
        class_idx_b: Second class index (Cardigan)
        layer_name: Layer to use for Grad-CAM
        
    Returns:
        cam_a: Grad-CAM for class A
        cam_b: Grad-CAM for class B
        diff_cam: Difference between CAMs (highlighting discriminative features)
    """
    # Initialize Grad-CAM
    grad_cam = GradCAM(model, layer_name)
    
    # Generate CAMs for both classes
    cam_a = grad_cam.generate(img_tensor, class_idx_a)
    cam_b = grad_cam.generate(img_tensor, class_idx_b)
    
    # Calculate difference CAM
    diff_cam = np.abs(cam_a - cam_b)
    diff_cam = (diff_cam - diff_cam.min()) / (diff_cam.max() - diff_cam.min() + 1e-8)
    
    return cam_a, cam_b, diff_cam


def visualize_contrastive_gradcam(img_tensor, cam_a, cam_b, diff_cam, 
                                 class_a_name="Pembroke", class_b_name="Cardigan"):
    """
    Visualize contrastive Grad-CAM results
    
    Args:
        img_tensor: Input image tensor
        cam_a: Grad-CAM for class A
        cam_b: Grad-CAM for class B
        diff_cam: Difference between CAMs
        class_a_name: Name of class A for display
        class_b_name: Name of class B for display
    """
    # Convert tensor to image
    img = img_tensor[0].permute(1, 2, 0).cpu().numpy()
    img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    img = img.astype(np.uint8)
    
    # Create heatmap overlays
    heatmap_a = cv2.applyColorMap(np.uint8(255 * cam_a), cv2.COLORMAP_JET)
    heatmap_b = cv2.applyColorMap(np.uint8(255 * cam_b), cv2.COLORMAP_JET)
    heatmap_diff = cv2.applyColorMap(np.uint8(255 * diff_cam), cv2.COLORMAP_JET)
    
    # Convert BGR to RGB
    heatmap_a = cv2.cvtColor(heatmap_a, cv2.COLOR_BGR2RGB)
    heatmap_b = cv2.cvtColor(heatmap_b, cv2.COLOR_BGR2RGB)
    heatmap_diff = cv2.cvtColor(heatmap_diff, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmaps on original image
    alpha = 0.4
    overlay_a = img * (1 - alpha) + heatmap_a * alpha
    overlay_b = img * (1 - alpha) + heatmap_b * alpha
    overlay_diff = img * (1 - alpha) + heatmap_diff * alpha
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(cam_a, cmap='jet')
    plt.title(f"Grad-CAM: {class_a_name}")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(cam_b, cmap='jet')
    plt.title(f"Grad-CAM: {class_b_name}")
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(overlay_a.astype(np.uint8))
    plt.title(f"Overlay: {class_a_name}")
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(overlay_b.astype(np.uint8))
    plt.title(f"Overlay: {class_b_name}")
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(overlay_diff.astype(np.uint8))
    plt.title(f"Contrastive Features")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def example_contrastive_gradcam(model, img_path, class_idx_pembroke=0, class_idx_cardigan=1, device="cpu"):
    """Example of using contrastive Grad-CAM on a test image"""
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
    
    # Generate contrastive Grad-CAM
    cam_pembroke, cam_cardigan, diff_cam = generate_contrastive_gradcam(
        model, img_tensor, class_idx_pembroke, class_idx_cardigan
    )
    
    # Visualize results
    print(f"Prediction: Pembroke: {pembroke_prob:.2%}, Cardigan: {cardigan_prob:.2%}")
    visualize_contrastive_gradcam(
        img_tensor, cam_pembroke, cam_cardigan, diff_cam,
        class_a_name="Pembroke", class_b_name="Cardigan"
    )
    
    return cam_pembroke, cam_cardigan, diff_cam 