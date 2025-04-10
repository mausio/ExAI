import copy
import numpy as np
import matplotlib.pyplot as plt
import cv2

# PyTorch imports
import torch
import torch.nn as nn

# Setup device for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# Section 10: GradCAM Implementation
# ============================================================================


class GradCAM:
    """
    GradCAM implementation for CNN visualization.

    This class implements the Gradient-weighted Class Activation Mapping (Grad-CAM)
    technique to visualize which parts of an image are important for a CNN's prediction.

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization", https://arxiv.org/abs/1610.02391
    """

    def __init__(self, model, target_layer):
        """
        Initializes GradCAM with a model and target layer

        Args:
            model: The trained PyTorch model
            target_layer: The convolutional layer to use for generating the CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.gradients = None
        self.activations = None
        self.register_hooks()
        self.model.eval()

    def register_hooks(self):
        """Registers forward and backward hooks to the target layer"""

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Register the hooks
        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

        # Store the handles for removal later
        self.hooks = [forward_handle, backward_handle]

    def remove_hooks(self):
        """Removes all registered hooks"""
        for hook in self.hooks:
            hook.remove()

    def __call__(self, input_tensor, target_class=None):
        """
        Generates the Grad-CAM for the input tensor

        Args:
            input_tensor: Input image (must be normalized the same way as training data)
            target_class: Target class index. If None, uses the predicted class.

        Returns:
            cam: The normalized Grad-CAM heatmap
        """
        # Forward pass
        input_tensor = input_tensor.to(device)

        # Zero gradients
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)

        # If target_class is None, use predicted class
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()

        # One-hot encoding of the target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1

        # Backward pass to get gradients
        output.backward(gradient=one_hot, retain_graph=True)

        # Get mean gradients and activations
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight the activations by the gradients
        for i in range(pooled_gradients.shape[0]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        # Average activations over the channel dimension
        cam = torch.mean(self.activations, dim=1).squeeze()

        # ReLU on the heatmap
        cam = torch.maximum(cam, torch.tensor(0.0).to(device))

        # Normalize the heatmap
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)

        # Resize to the input image size
        cam = cam.cpu().numpy()

        return cam


def apply_gradcam(model, img_tensor, img_np, target_class=None, layer_name="layer4"):
    """
    Applies GradCAM to visualize model attention

    Args:
        model: Trained PyTorch model
        img_tensor: Input image tensor (1, C, H, W)
        img_np: Original numpy image for visualization (RGB)
        target_class: Target class for visualization
        layer_name: Name of layer to use for GradCAM (default: 'layer4')

    Returns:
        visualization: Heatmap overlaid on original image
        cam: Raw heatmap
    """
    # Get the target layer
    target_layer = model.layer4

    # Create GradCAM instance
    grad_cam = GradCAM(model, target_layer)

    # Generate heatmap
    cam = grad_cam(img_tensor, target_class)

    # Resize CAM to input image size
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))

    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

    # Convert to RGB (from BGR)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay heatmap on original image
    alpha = 0.4
    visualization = heatmap * alpha + img_np * (1 - alpha)
    visualization = np.uint8(visualization)

    # Remove hooks
    grad_cam.remove_hooks()

    return visualization, cam


def visualize_gradcam(model, dataloader, class_names, num_images=5):
    """
    Visualizes GradCAM for a batch of images

    Args:
        model: Trained PyTorch model
        dataloader: DataLoader containing images to visualize
        class_names: Names of the classes
        num_images: Number of images to visualize
    """
    # Set model to evaluation mode
    model.eval()

    # Get a batch of images
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]

    # Create a figure
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

    for i, (image, label) in enumerate(zip(images, labels)):
        # Convert to numpy image for display
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(
            img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]),
            0,
            1,
        )

        # Prepare input for model
        input_tensor = image.unsqueeze(0).to(device)

        # Get model prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            prob = torch.nn.functional.softmax(output, dim=1)

        # Generate GradCAM for true class
        true_cam, _ = apply_gradcam(model, input_tensor, img_np, label.item())

        # Generate GradCAM for predicted class
        pred_cam, _ = apply_gradcam(model, input_tensor, img_np, pred.item())

        # Display original image
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(
            f"True: {class_names[label]}\nPred: {class_names[pred]} ({prob[0][pred.item()]:.2f})"
        )
        axes[i, 0].axis("off")

        # Display GradCAM for true class
        axes[i, 1].imshow(true_cam)
        axes[i, 1].set_title(f"GradCAM for {class_names[label]}")
        axes[i, 1].axis("off")

        # Display GradCAM for predicted class
        axes[i, 2].imshow(pred_cam)
        axes[i, 2].set_title(f"GradCAM for {class_names[pred]}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("gradcam_visualizations.png")
    plt.show()


# ============================================================================
# Section 11: Layer-wise Relevance Propagation (LRP) Implementation
# ============================================================================


class LRP:
    """
    Simplified Layer-wise Relevance Propagation (LRP) for CNN visualization.
    
    This implementation uses a simpler gradient-based approach that avoids
    the issues with tensor views and in-place modifications.
    """
    
    def __init__(self, model, epsilon=1e-9):
        """
        Initializes LRP with a model
        
        Args:
            model: The trained PyTorch model (ResNet50)
            epsilon: Small constant for numerical stability
        """
        self.model = model
        self.epsilon = epsilon
        self.model.eval()
    
    def __call__(self, input_tensor, target_class=None):
        """
        Generates the LRP heatmap for the input tensor using a simplified approach
        
        Args:
            input_tensor: Input image tensor (must be normalized)
            target_class: Target class index. If None, uses the predicted class.
            
        Returns:
            relevance_map: The normalized LRP heatmap
        """
        # Make a detached copy of the input that requires gradient
        input_copy = input_tensor.clone().detach().to(device)
        input_copy.requires_grad = True
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_copy)
        
        # If target_class is None, use predicted class
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
        
        # One-hot encoding for the target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        
        # Backward pass to get gradients
        output.backward(gradient=one_hot)
        
        # Get the gradient with respect to the input
        # This represents how much each input pixel affects the output
        grad = input_copy.grad.clone()
        
        # Element-wise product of input and gradient
        # This gives us a relevance map highlighting important features
        relevance = (input_copy * grad).sum(dim=1).squeeze()
        
        # Take absolute value and normalize
        relevance = torch.abs(relevance)
        if torch.max(relevance) > 0:
            relevance = relevance / torch.max(relevance)
        
        return relevance.detach().cpu().numpy()

def apply_lrp(model, img_tensor, img_np, target_class=None):
    """
    Applies simplified LRP to visualize model contributions
    
    Args:
        model: Trained PyTorch model
        img_tensor: Input image tensor (1, C, H, W)
        img_np: Original numpy image for visualization (RGB)
        target_class: Target class for visualization
        
    Returns:
        visualization: Heatmap overlaid on original image
        relevance_map: Raw relevance map
    """
    # Create LRP instance
    lrp = LRP(model)
    
    try:
        # Generate relevance map
        relevance_map = lrp(img_tensor, target_class)
        
        if relevance_map is None:
            # Return a blank heatmap if LRP fails
            relevance_map = np.zeros((img_np.shape[0], img_np.shape[1]))
            visualization = img_np.copy()
            return visualization, relevance_map
        
        # Resize relevance map to input image size
        relevance_resized = cv2.resize(relevance_map, (img_np.shape[1], img_np.shape[0]))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * relevance_resized), cv2.COLORMAP_JET)
        
        # Convert to RGB (from BGR)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        alpha = 0.4
        visualization = heatmap * alpha + img_np * (1 - alpha)
        visualization = np.uint8(visualization)
        
        return visualization, relevance_map
    except Exception as e:
        print(f"Error during LRP computation: {e}")
        # Return a blank heatmap if LRP fails
        relevance_map = np.zeros((img_np.shape[0], img_np.shape[1]))
        visualization = img_np.copy()
        return visualization, relevance_map


def visualize_lrp(model, dataloader, class_names, num_images=5):
    """
    Visualizes LRP for a batch of images

    Args:
        model: Trained PyTorch model
        dataloader: DataLoader containing images to visualize
        class_names: Names of the classes
        num_images: Number of images to visualize
    """
    # TODO: Display LRP per Layer
    # Set model to evaluation mode
    model.eval()

    # Get a batch of images
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]

    # Create a figure
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

    for i, (image, label) in enumerate(zip(images, labels)):
        # Convert to numpy image for display
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(
            img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]),
            0,
            1,
        )

        # Prepare input for model
        input_tensor = image.unsqueeze(0).to(device)

        # Get model prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            prob = torch.nn.functional.softmax(output, dim=1)

        # Generate LRP for true class
        true_lrp, _ = apply_lrp(model, input_tensor, img_np, label.item())
        
        # Generate LRP for predicted class
        pred_lrp, _ = apply_lrp(model, input_tensor, img_np, pred.item())

        # Display original image
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(
            f"True: {class_names[label]}\nPred: {class_names[pred]} ({prob[0][pred.item()]:.2f})"
        )
        axes[i, 0].axis("off")

        # Display LRP for true class
        axes[i, 1].imshow(true_lrp)
        axes[i, 1].set_title(f"LRP for {class_names[label]}")
        axes[i, 1].axis("off")

        # Display LRP for predicted class
        axes[i, 2].imshow(pred_lrp)
        axes[i, 2].set_title(f"LRP for {class_names[pred]}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("lrp_visualizations.png")
    plt.show()


# ============================================================================
# Section 12: XAI Methods Comparison
# ============================================================================


def compare_xai_methods(model, dataloader, class_names, num_images=3):
    """
    Visually compares different XAI methods on the same images

    Args:
        model: Trained PyTorch model
        dataloader: DataLoader containing images to visualize
        class_names: Names of the classes
        num_images: Number of images to visualize
    """
    # Set model to evaluation mode
    model.eval()

    # Get a batch of images
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]

    # Create a figure
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

    for i, (image, label) in enumerate(zip(images, labels)):
        # Convert to numpy image for display
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(
            img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]),
            0,
            1,
        )

        # Prepare input for model
        input_tensor = image.unsqueeze(0).to(device)

        # Get model prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            prob = torch.nn.functional.softmax(output, dim=1)

        # Generate GradCAM for predicted class
        gradcam_vis, _ = apply_gradcam(model, input_tensor, img_np, pred.item())

        # Generate LRP for predicted class
        lrp_vis, _ = apply_lrp(model, input_tensor, img_np, pred.item())

        # Handle single image case (no row dimension in axes)
        if num_images == 1:
            ax0, ax1, ax2 = axes
        else:
            ax0, ax1, ax2 = axes[i]

        # Display original image
        ax0.imshow(img_np)
        ax0.set_title(
            f"Original\nTrue: {class_names[label]}\nPred: {class_names[pred]} ({prob[0][pred.item()]:.2f})"
        )
        ax0.axis("off")

        # Display GradCAM
        ax1.imshow(gradcam_vis)
        ax1.set_title("GradCAM")
        ax1.axis("off")

        # Display LRP
        ax2.imshow(lrp_vis)
        ax2.set_title("Layer-wise Relevance Propagation")
        ax2.axis("off")

    plt.tight_layout()
    plt.savefig("xai_comparison.png")
    plt.show()

    # Add a detailed analysis
    print("\nXAI Methods Comparison Analysis:")
    print("-------------------------------")
    print("GradCAM:")
    print("  - Highlights regions that most strongly activate the target class.")
    print(
        "  - Focuses on the last convolutional layer, which may not capture fine details."
    )
    print("  - Computationally efficient and easy to implement.")
    print("  - Provides a coarse localization of important features.")

    print("\nLayer-wise Relevance Propagation (LRP):")
    print("  - Propagates relevance scores from output to input through all layers.")
    print("  - Can provide finer details about feature importance.")
    print("  - More computationally intensive than GradCAM.")
    print("  - Better at capturing feature interactions across multiple layers.")

    print("\nComparison:")
    print("  - GradCAM tends to highlight broader regions.")
    print("  - LRP often produces more detailed and precise feature attributions.")
    print(
        "  - For complex features (like dog breeds), these visualizations help identify"
    )
    print("    which visual traits the model is using to distinguish between classes.")

def compare_gradcam_classes(model, dataloader, class_names, num_images=1):
    """
    Compares GradCAM visualizations for different classes (Pembroke vs Cardigan)
    on the same image.
    
    Args:
        model: Trained PyTorch model
        dataloader: DataLoader containing images to visualize
        class_names: Names of the classes (expecting Pembroke and Cardigan)
        num_images: Number of images to visualize
    """
    # Set model to evaluation mode
    model.eval()
    
    # Get a batch of images
    images, labels = next(iter(dataloader))
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Verify we have both classes in class_names
    if 'Pembroke' not in class_names or 'Cardigan' not in class_names:
        print("Error: Class names must include both 'Pembroke' and 'Cardigan'")
        return
    
    pembroke_idx = class_names.index('Pembroke')
    cardigan_idx = class_names.index('Cardigan')
    
    # Create a figure
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    
    for i, (image, label) in enumerate(zip(images, labels)):
        # Convert to numpy image for display
        img_np = image.cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(
            img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]),
            0,
            1,
        )
        
        # Prepare input for model
        input_tensor = image.unsqueeze(0).to(device)
        
        # Get model prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            prob = torch.nn.functional.softmax(output, dim=1)
            
            # Get probabilities for each class
            pembroke_prob = prob[0][pembroke_idx].item()
            cardigan_prob = prob[0][cardigan_idx].item()
        
        # Generate GradCAM for Pembroke class
        pembroke_gradcam, _ = apply_gradcam(model, input_tensor, img_np, pembroke_idx)
        
        # Generate GradCAM for Cardigan class
        cardigan_gradcam, _ = apply_gradcam(model, input_tensor, img_np, cardigan_idx)
        
        # Handle single image case (no row dimension in axes)
        if num_images == 1:
            ax0, ax1, ax2 = axes
        else:
            ax0, ax1, ax2 = axes[i]
        
        # Display original image
        ax0.imshow(img_np)
        ax0.set_title(
            f"Pred: {class_names[pred]} ({prob[0][pred.item()]:.2f})\n"
            f"Pembroke: {pembroke_prob:.2f}, Cardigan: {cardigan_prob:.2f}"
        )
        ax0.axis("off")
        
        # Display GradCAM for Pembroke
        ax1.imshow(pembroke_gradcam)
        ax1.set_title(f"GradCAM for Pembroke")
        ax1.axis("off")
        
        # Display GradCAM for Cardigan
        ax2.imshow(cardigan_gradcam)
        ax2.set_title(f"GradCAM for Cardigan")
        ax2.axis("off")
    
    plt.tight_layout()
    plt.savefig("gradcam_class_comparison.png")
    plt.show()
