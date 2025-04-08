import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class LRPHook:
    """
    Layer-wise Relevance Propagation hook for PyTorch modules
    Based on the paper: Layer-wise Relevance Propagation: An Overview
    """
    def __init__(self):
        self.forward_values = {}
        self.hooks = []
        
    def register_hooks(self, model):
        """Register hooks for all modules in the model"""
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.AvgPool2d, nn.MaxPool2d)):
                hook = module.register_forward_hook(self._forward_hook(name))
                self.hooks.append(hook)
    
    def _forward_hook(self, name):
        """Hook function to store forward pass values"""
        def hook(module, input, output):
            self.forward_values[name] = (input[0].detach(), output.detach())
        return hook
    
    def remove_hooks(self):
        """Remove all hooks to prevent memory leaks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class LRP:
    """
    Layer-wise Relevance Propagation for CNN visualization
    """
    def __init__(self, model, epsilon=1e-9):
        self.model = model
        self.model.eval()
        self.epsilon = epsilon
        self.hook = LRPHook()
        self.hook.register_hooks(self.model)
        
    def __del__(self):
        """Clean up when object is deleted"""
        try:
            self.hook.remove_hooks()
        except:
            pass
        
    def generate(self, input_tensor, target_class=None):
        """
        Generate LRP heatmap for the input image
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Class index to generate LRP for. If None, uses the max prediction
            
        Returns:
            heatmap: LRP relevance map (H, W)
        """
        # Forward pass
        output = self.model(input_tensor)
        
        # Get target class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # Initialize relevance with one-hot vector
        relevance = torch.zeros_like(output)
        relevance[0, target_class] = output[0, target_class]
        
        # Get layers in reverse order
        modules = list(self.model.modules())
        layer_names = list(self.hook.forward_values.keys())
        
        # Perform LRP in reverse order through the network
        for i in range(len(layer_names) - 1, -1, -1):
            name = layer_names[i]
            for module_name, module in self.model.named_modules():
                if module_name == name:
                    # Get forward values
                    input_tensor, output_tensor = self.hook.forward_values[name]
                    
                    # LRP for different layer types
                    if isinstance(module, nn.Conv2d):
                        relevance = self._lrp_conv(module, input_tensor, output_tensor, relevance)
                    elif isinstance(module, nn.Linear):
                        relevance = self._lrp_linear(module, input_tensor, output_tensor, relevance)
                    elif isinstance(module, (nn.AvgPool2d, nn.MaxPool2d)):
                        relevance = self._lrp_pool(module, input_tensor, output_tensor, relevance)
                    break
        
        # Sum over channels and normalize
        relevance_map = relevance.sum(dim=1)[0].detach().cpu().numpy()
        
        # Apply ReLU to focus on positive contributions
        relevance_map = np.maximum(relevance_map, 0)
        
        # Normalize
        relevance_map = (relevance_map - relevance_map.min()) / (relevance_map.max() - relevance_map.min() + 1e-9)
        
        return relevance_map
    
    def _lrp_conv(self, layer, input_tensor, output_tensor, relevance):
        """LRP for convolutional layers using epsilon rule"""
        # Get weights and reshape
        weights = layer.weight
        
        # Forward pass with epsilon stabilization
        z = torch.nn.functional.conv2d(input_tensor, weights, bias=layer.bias,
                                      stride=layer.stride, padding=layer.padding,
                                      dilation=layer.dilation, groups=layer.groups)
        z += self.epsilon * ((z >= 0).float() * 2 - 1)
        
        # Compute relevance contribution
        s = (relevance / z).data
        c = torch.nn.functional.conv_transpose2d(s, weights, stride=layer.stride,
                                               padding=layer.padding, groups=layer.groups)
        relevance = input_tensor * c
        
        return relevance
    
    def _lrp_linear(self, layer, input_tensor, output_tensor, relevance):
        """LRP for fully connected layers using epsilon rule"""
        # Get weights
        weights = layer.weight
        
        # Forward pass with epsilon stabilization
        z = torch.matmul(input_tensor, weights.t())
        if layer.bias is not None:
            z += layer.bias.unsqueeze(0).expand_as(z)
        z += self.epsilon * ((z >= 0).float() * 2 - 1)
        
        # Compute relevance contribution
        s = (relevance / z).data
        c = torch.matmul(s, weights)
        relevance = input_tensor * c
        
        return relevance
    
    def _lrp_pool(self, layer, input_tensor, output_tensor, relevance):
        """LRP for pooling layers (simple redistribution)"""
        if isinstance(layer, nn.AvgPool2d):
            # For average pooling, evenly redistribute
            return torch.nn.functional.interpolate(relevance, size=input_tensor.shape[2:])
        else:
            # For max pooling (simple redistribution)
            return torch.nn.functional.interpolate(relevance, size=input_tensor.shape[2:])


def generate_lrp_heatmap(model, img_tensor, target_class=None):
    """
    Generate LRP heatmap for the input image
    
    Args:
        model: PyTorch model
        img_tensor: Input image tensor
        target_class: Class index to generate LRP for. If None, uses the max prediction
        
    Returns:
        relevance_map: LRP relevance map
    """
    lrp = LRP(model)
    relevance_map = lrp.generate(img_tensor, target_class)
    return relevance_map


def visualize_lrp(img_tensor, relevance_map, class_name=""):
    """
    Visualize LRP relevance map
    
    Args:
        img_tensor: Input image tensor
        relevance_map: LRP relevance map
        class_name: Name of the class for display
    """
    # Convert tensor to image
    img = img_tensor[0].permute(1, 2, 0).cpu().numpy()
    img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    img = img.astype(np.uint8)
    
    # Create heatmap overlay
    import cv2
    heatmap = cv2.applyColorMap(np.uint8(255 * relevance_map), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmap on original image
    alpha = 0.4
    overlay = img * (1 - alpha) + heatmap * alpha
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(relevance_map, cmap='jet')
    plt.title(f"LRP Heatmap{': ' + class_name if class_name else ''}")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay.astype(np.uint8))
    plt.title("Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def example_lrp(model, img_path, class_idx_pembroke=0, class_idx_cardigan=1, device="cpu"):
    """Example of using LRP on a test image"""
    # Preprocessing function
    from torchvision import transforms
    import torch.nn.functional as F
    from PIL import Image
    
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
    
    # Get the predicted class
    pred_class = outputs.argmax(dim=1).item()
    class_name = "Pembroke" if pred_class == class_idx_pembroke else "Cardigan"
    
    # Generate LRP heatmap for the predicted class
    relevance_map = generate_lrp_heatmap(model, img_tensor, pred_class)
    
    # Visualize results
    print(f"Prediction: Pembroke: {pembroke_prob:.2%}, Cardigan: {cardigan_prob:.2%}")
    visualize_lrp(img_tensor, relevance_map, class_name)
    
    return relevance_map


def compare_gradcam_lrp(model, img_path, class_idx=None, class_idx_pembroke=0, class_idx_cardigan=1, device="cpu"):
    """
    Compare Grad-CAM and LRP visualizations for the same image
    
    Args:
        model: PyTorch model
        img_path: Path to the input image
        class_idx: Class index to visualize. If None, uses the predicted class
        class_idx_pembroke: Index for Pembroke class
        class_idx_cardigan: Index for Cardigan class
        device: Device to run the model on
    """
    # Import necessary modules
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image
    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Preprocessing
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
        pred_class = outputs.argmax(dim=1).item() if class_idx is None else class_idx
        
    # Get class name
    class_name = "Pembroke" if pred_class == class_idx_pembroke else "Cardigan"
    
    # Generate Grad-CAM and LRP visualizations
    from xai_gradcam import GradCAM
    gradcam = GradCAM(model)
    cam = gradcam.generate(img_tensor, pred_class)
    
    lrp = LRP(model)
    lrp_map = lrp.generate(img_tensor, pred_class)
    
    # Convert tensor to image
    img = img_tensor[0].permute(1, 2, 0).cpu().numpy()
    img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    img = img.astype(np.uint8)
    
    # Create heatmap overlays
    gradcam_heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    lrp_heatmap = cv2.applyColorMap(np.uint8(255 * lrp_map), cv2.COLORMAP_JET)
    
    # Convert BGR to RGB
    gradcam_heatmap = cv2.cvtColor(gradcam_heatmap, cv2.COLOR_BGR2RGB)
    lrp_heatmap = cv2.cvtColor(lrp_heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay heatmaps on original image
    alpha = 0.4
    gradcam_overlay = img * (1 - alpha) + gradcam_heatmap * alpha
    lrp_overlay = img * (1 - alpha) + lrp_heatmap * alpha
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(img)
    plt.title(f"Original Image - {class_name} ({probs[pred_class]:.2%})")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(cam, cmap='jet')
    plt.title(f"Grad-CAM Heatmap")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(lrp_map, cmap='jet')
    plt.title(f"LRP Heatmap")
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.imshow(gradcam_overlay.astype(np.uint8))
    plt.title(f"Grad-CAM Overlay")
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    plt.imshow(lrp_overlay.astype(np.uint8))
    plt.title(f"LRP Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return cam, lrp_map 