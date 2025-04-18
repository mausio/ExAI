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


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.gradients = None
        self.activations = None
        self.register_hooks()
        self.model.eval()

    def register_hooks(self):
        def forward_hook(module, input, output):
            # Store the activations of target layer during forward pass
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            # Store the gradients of target layer during backward pass
            self.gradients = grad_output[0].detach()

        forward_handle = self.target_layer.register_forward_hook(forward_hook)
        backward_handle = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks = [forward_handle, backward_handle]

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def __call__(self, input_tensor, target_class=None):
        input_tensor = input_tensor.to(device)
        # Reset gradients
        self.model.zero_grad()

        # => Forward pass 
        output = self.model(input_tensor)

        if target_class is None:
            # ..use predicted class..
            target_class = torch.argmax(output, dim=1).item()

        # One-hot encoding in sparse bitmap for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1

        # <= Backward pass to get gradients
        output.backward(gradient=one_hot, retain_graph=True)
        # ..backward hooks are called here to store in gradients.
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        # .. contains mean gradients and activations.

        # Weight the activations by the gradients
        for i in range(pooled_gradients.shape[0]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        avg_activations = torch.mean(self.activations, dim=1).squeeze()
        # ..over the channel dimension.

        # ReLU on the heatmap
        heatmap = torch.maximum(avg_activations, torch.tensor(0.0).to(device))

        # Normalize heatmap
        if torch.max(heatmap) > 0:
            heatmap = heatmap / torch.max(heatmap)

        return heatmap.cpu().numpy()


def apply_gradcam(model, img_tensor, img_np, target_class=None, layer_name="layer4"):
    # Get the target layer
    target_layer = model.layer4

    # Create GradCAM instance
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam(img_tensor, target_class)

    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    # ..to input image size

    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    # ..and to RGB (from BGR)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay heatmap on original image
    alpha = 0.4
    visualization = heatmap * alpha + img_np * (1 - alpha)
    visualization = np.uint8(visualization)

    # Remove hooks for possible reuse
    grad_cam.remove_hooks()

    return visualization, cam


def visualize_gradcam(model, dataloader, class_names, num_images=5):
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
        axes[i, 1].set_title(f"GradCAM for true {class_names[label]}")
        axes[i, 1].axis("off")

        # Display GradCAM for predicted class
        axes[i, 2].imshow(pred_cam)
        axes[i, 2].set_title(f"GradCAM for predicted {class_names[pred]}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("gradcam_visualizations.png")
    plt.show()


class LRP:
    def __init__(self, model, epsilon=1e-9):
        self.model = model
        self.epsilon = epsilon
        self.model.eval()
    
    def __call__(self, input_tensor, target_class=None):
        # Make a detached copy of the input
        input_copy = input_tensor.clone().detach().to(device)
        input_copy.requires_grad = True
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_copy)
        
        if target_class is None:
            # ..use predicted class
            target_class = torch.argmax(output, dim=1).item()
        
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
        
        # Overlay heatmap on original numpy array (image)
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
        axes[i, 1].set_title(f"LRP for true {class_names[label]}")
        axes[i, 1].axis("off")

        # Display LRP for predicted class
        axes[i, 2].imshow(pred_lrp)
        axes[i, 2].set_title(f"LRP for predicted {class_names[pred]}")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("lrp_visualizations.png")
    plt.show()


def compare_xai_methods(model, dataloader, class_names, num_images=3):
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
    model.eval()  # Set model to evaluation mode
    
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

def compare_lrp_classes(model, dataloader, class_names, num_images=1):
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
        
        # Generate LRP for Pembroke class
        pembroke_lrp, _ = apply_lrp(model, input_tensor, img_np, pembroke_idx)
        
        # Generate LRP for Cardigan class
        cardigan_lrp, _ = apply_lrp(model, input_tensor, img_np, cardigan_idx)
        
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
        
        # Display LRP for Pembroke
        ax1.imshow(pembroke_lrp)
        ax1.set_title(f"LRP for Pembroke")
        ax1.axis("off")
        
        # Display LRP for Cardigan
        ax2.imshow(cardigan_lrp)
        ax2.set_title(f"LRP for Cardigan")
        ax2.axis("off")
    
    plt.tight_layout()
    plt.savefig("lrp_class_comparison.png")
    plt.show()
