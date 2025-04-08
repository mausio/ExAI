#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corgi Classifier - ResNet50 Fine-tuning for Distinguishing Welsh Corgi Breeds
============================================================================

This script demonstrates fine-tuning a ResNet50 model on the Stanford Dogs Dataset 
to distinguish between Pembroke and Cardigan Welsh Corgis.

The code is structured similar to a Jupyter notebook for easy transfer.
"""

# ============================================================================
# Section 1: Setup and Imports
# ============================================================================

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image
import cv2

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms, models
from torchvision.datasets.utils import download_url, extract_archive

# For evaluation metrics
from sklearn.metrics import confusion_matrix, classification_report

# Setup device for training
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# Section 2: Dataset Downloading and Extraction
# ============================================================================

def download_and_extract_dataset(download_dir, extract_dir):
    """
    Downloads and extracts the Stanford Dogs Dataset
    """
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    
    # Download the dataset
    dataset_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    filename = os.path.basename(dataset_url)
    filepath = os.path.join(download_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        download_url(dataset_url, download_dir)
    else:
        print(f"File {filename} already exists in {download_dir}")
    
    # Extract the dataset
    if not os.path.exists(os.path.join(extract_dir, "Images")):
        print(f"Extracting {filename} to {extract_dir}...")
        extract_archive(filepath, extract_dir)
    else:
        print(f"Dataset already extracted to {extract_dir}")

# ============================================================================
# Section 3: Custom Dataset Classes
# ============================================================================

class CorgiDataset(Dataset):
    def __init__(self, dataset_root, transform=None):
        """
        Custom dataset for Pembroke and Cardigan Welsh Corgis
        
        Args:
            dataset_root: Root directory containing the Images folder
            transform: PyTorch transforms for data augmentation
        """
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = ['Pembroke', 'Cardigan']
        
        # Find the corgi breed directories
        images_dir = os.path.join(dataset_root, "Images")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found at {images_dir}")
            
        all_breeds = os.listdir(images_dir)
        
        pembroke_dir = None
        cardigan_dir = None
        
        for breed in all_breeds:
            if "Pembroke" in breed:
                pembroke_dir = os.path.join(images_dir, breed)
            elif "Cardigan" in breed:
                cardigan_dir = os.path.join(images_dir, breed)
        
        if not pembroke_dir or not cardigan_dir:
            raise ValueError("Could not find Pembroke or Cardigan directories")
        
        print(f"Pembroke directory: {pembroke_dir}")
        print(f"Cardigan directory: {cardigan_dir}")
        
        # Load Pembroke images (label 0)
        for img_name in os.listdir(pembroke_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(pembroke_dir, img_name))
                self.labels.append(0)  # Pembroke
        
        # Load Cardigan images (label 1)
        for img_name in os.listdir(cardigan_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(cardigan_dir, img_name))
                self.labels.append(1)  # Cardigan
        
        # Print dataset statistics
        print(f"Total number of images: {len(self.image_paths)}")
        print(f"Pembroke images: {sum(1 for label in self.labels if label == 0)}")
        print(f"Cardigan images: {sum(1 for label in self.labels if label == 1)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image and the same label
            blank_image = torch.zeros((3, 224, 224)) if self.transform else Image.new('RGB', (224, 224), (0, 0, 0))
            return blank_image, self.labels[idx]

class TransformedSubset(Dataset):
    """Wrapper for applying transforms to a subset of the dataset"""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, idx):
        # Correctly handle the subset indexing
        image, label = self.subset.dataset[self.subset.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.subset)

# ============================================================================
# Section 4: Data Preparation and Loaders
# ============================================================================

def prepare_dataloaders(dataset_root, batch_size=32, num_workers=2):
    """
    Prepares the dataloaders for training and validation
    
    Args:
        dataset_root: Root directory containing the Images folder
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        class_names: Names of the classes
    """
    # Data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    # Create the dataset
    dataset = CorgiDataset(dataset_root, transform=None)  # No transform yet
    
    # Split into training and validation sets (80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create new datasets with appropriate transforms
    train_dataset_transformed = TransformedSubset(train_dataset, data_transforms['train'])
    val_dataset_transformed = TransformedSubset(val_dataset, data_transforms['val'])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset_transformed, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset_transformed, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Training set size: {len(train_dataset)} images")
    print(f"Validation set size: {len(val_dataset)} images")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader, dataset.class_names

# ============================================================================
# Section 5: Model Setup
# ============================================================================

def setup_model(num_classes=2):
    """
    Sets up the ResNet50 model for fine-tuning
    
    Args:
        num_classes: Number of output classes
        
    Returns:
        model: The ResNet50 model configured for fine-tuning
    """
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=True)
    
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze later layers for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    # Move model to device
    model = model.to(device)
    
    # Print model information
    print(f"ResNet50 model configured for {num_classes} classes")
    print(f"Trainable parameters in model.layer4: {sum(p.numel() for p in model.layer4.parameters() if p.requires_grad)}")
    print(f"Trainable parameters in model.fc: {sum(p.numel() for p in model.fc.parameters() if p.requires_grad)}")
    
    return model

# ============================================================================
# Section 6: Training Functions
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer):
    """
    Trains the model for one epoch
    
    Args:
        model: The model to train
        dataloader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        
    Returns:
        epoch_loss: Average loss for the epoch
        epoch_acc: Accuracy for the epoch
    """
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_corrects = 0
    
    # Iterate over data
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        # Backward + optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
    return epoch_loss, epoch_acc.item()

def validate_epoch(model, dataloader, criterion):
    """
    Validates the model for one epoch
    
    Args:
        model: The model to validate
        dataloader: DataLoader for validation data
        criterion: Loss function
        
    Returns:
        epoch_loss: Average loss for the epoch
        epoch_acc: Accuracy for the epoch
    """
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0
    
    # Iterate over data
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
    return epoch_loss, epoch_acc.item()

def train_model(model, train_loader, val_loader, num_epochs=15, patience=5):
    """
    Trains the model and tracks performance metrics
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of epochs to train for
        patience: Number of epochs to wait for improvement before early stopping
        
    Returns:
        model: The trained model
        history: Dictionary containing loss and accuracy history
    """
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    no_improve_epochs = 0
    
    # History for tracking metrics
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 40)
        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation phase
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Deep copy the model if best accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        print(f'Best val Acc: {best_acc:.4f}')
        
        # Early stopping
        if no_improve_epochs >= patience:
            print(f'Early stopping after {epoch+1} epochs without improvement')
            break
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

# ============================================================================
# Section 7: Evaluation Functions
# ============================================================================

def evaluate_model(model, dataloader, class_names):
    """
    Evaluates the model on a dataset and computes metrics
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for the evaluation data
        class_names: Names of the classes
        
    Returns:
        y_true: True labels
        y_pred: Predicted labels
        report: Classification report
    """
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    # Print classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    return y_true, y_pred, report

def plot_training_history(history):
    """
    Plots the training history
    
    Args:
        history: Dictionary containing loss and accuracy history
    """
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# ============================================================================
# Section 8: Model Saving and Loading
# ============================================================================

def save_model(model, save_path, class_names, optimizer=None, epoch=None, history=None):
    """
    Saves the model to a file in multiple formats
    
    Args:
        model: The model to save
        save_path: Path to save the model
        class_names: Names of the classes
        optimizer: Optimizer to save (optional)
        epoch: Current epoch (optional)
        history: Training history (optional)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save complete model for inference
    torch.save(model, save_path.replace('.pth', '_full.pth'))
    print(f"Complete model saved to: {save_path.replace('.pth', '_full.pth')}")
    
    # Save model weights only
    torch.save(model.state_dict(), save_path.replace('.pth', '_weights.pth'))
    print(f"Model weights saved to: {save_path.replace('.pth', '_weights.pth')}")
    
    # Save checkpoint with additional information
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'classes': class_names,
    }
    
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
        
    if history:
        checkpoint['history'] = history
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to: {save_path}")
    
    # Export to ONNX format for deployment
    try:
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        torch.onnx.export(
            model,
            dummy_input,
            save_path.replace('.pth', '.onnx'),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"ONNX model saved to: {save_path.replace('.pth', '.onnx')}")
    except Exception as e:
        print(f"Error exporting to ONNX format: {e}")

def load_model(load_path, model=None):
    """
    Loads a model from a file
    
    Args:
        load_path: Path to load the model from
        model: Model to load weights into (optional)
        
    Returns:
        model: The loaded model
        checkpoint: The loaded checkpoint
    """
    try:
        checkpoint = torch.load(load_path, map_location=device)
        
        if model is None:
            # Try loading the full model
            if load_path.endswith('_full.pth'):
                model = torch.load(load_path, map_location=device)
                print(f"Full model loaded from: {load_path}")
                return model, None
            
            # Otherwise create a new model
            model = setup_model()
        
        # Load state dict if it exists
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"Model weights loaded from: {load_path}")
        return model, checkpoint
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# ============================================================================
# Section 9: Main Execution
# ============================================================================

def main():
    """Main execution function"""
    # Directory setup
    download_dir = "/content/drive/MyDrive/xAI-Corgis" # TODO: Change to local directory
    extract_dir = "/content/dogs" # TODO: Change to local directory 
    
    # Download and extract dataset
    download_and_extract_dataset(download_dir, extract_dir)
    
    # Prepare dataloaders
    train_loader, val_loader, class_names = prepare_dataloaders(extract_dir, batch_size=32)
    
    # Setup model
    model = setup_model(num_classes=len(class_names))
    
    # Check if model exists and load it
    os.makedirs(download_dir, exist_ok=True)
    model_path = os.path.join(download_dir, 'resnet50_corgi_classifier.pth')
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        model, checkpoint = load_model(model_path)
    else:
        # Train model
        print("Training a new model...")
        model, history = train_model(model, train_loader, val_loader, num_epochs=15)
        
        # Plot training history
        plot_training_history(history)
        
        # Evaluate model
        y_true, y_pred, report = evaluate_model(model, val_loader, class_names)
        
        # Save model
        save_model(
            model, 
            model_path,
            class_names=class_names,
            history=history
        )

    # Apply XAI methods
    print("\n" + "="*50)
    print("Applying XAI Methods for Model Interpretability")
    print("="*50)
    
    # Visualize with GradCAM
    print("\nGenerating GradCAM visualizations...")
    visualize_gradcam(model, val_loader, class_names, num_images=5)
    
    # Visualize with LRP
    print("\nGenerating Layer-wise Relevance Propagation visualizations...")
    visualize_lrp(model, val_loader, class_names, num_images=5)
    
    # Compare both XAI methods
    print("\nComparing GradCAM and LRP methods...")
    compare_xai_methods(model, val_loader, class_names, num_images=3)
    
    print("\nXAI visualization complete. All results saved as PNG files.")

if __name__ == "__main__":
    main()

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

def apply_gradcam(model, img_tensor, img_np, target_class=None, layer_name='layer4'):
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
        img_np = np.clip(img_np * np.array([0.229, 0.224, 0.225]) + 
                        np.array([0.485, 0.456, 0.406]), 0, 1)
        
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
        axes[i, 0].set_title(f"True: {class_names[label]}\nPred: {class_names[pred]} ({prob[0][pred.item()]:.2f})")
        axes[i, 0].axis('off')
        
        # Display GradCAM for true class
        axes[i, 1].imshow(true_cam)
        axes[i, 1].set_title(f"GradCAM for {class_names[label]}")
        axes[i, 1].axis('off')
        
        # Display GradCAM for predicted class
        axes[i, 2].imshow(pred_cam)
        axes[i, 2].set_title(f"GradCAM for {class_names[pred]}")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_visualizations.png')
    plt.show()

# ============================================================================
# Section 11: Layer-wise Relevance Propagation (LRP) Implementation
# ============================================================================

class LRP:
    """
    Layer-wise Relevance Propagation (LRP) for CNN visualization.
    
    This class implements the LRP technique to visualize which parts of an image
    contribute most to a model's prediction.
    
    Reference: Bach et al., "On Pixel-Wise Explanations for Non-Linear Classifier 
    Decisions by Layer-Wise Relevance Propagation", https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140
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
        
    def _clone_module(self, module, memo=None):
        """Create a copy of a module by recursively cloning its parameters and buffers"""
        if memo is None:
            memo = {}
        if id(module) in memo:
            return memo[id(module)]
        
        clone = copy.deepcopy(module)
        memo[id(module)] = clone
        
        return clone
    
    def _register_hooks(self, module, activations, relevances):
        """Register forward and backward hooks for LRP"""
        forward_hooks = []
        backward_hooks = []
        
        def forward_hook(m, input, output):
            activations[id(m)] = output.detach()
            
        def backward_hook(m, grad_in, grad_out):
            """Modified backward pass for LRP"""
            if id(m) in activations:
                with torch.no_grad():
                    # Get the activations from the forward pass
                    a = activations[id(m)]
                    if isinstance(m, nn.Conv2d):
                        # For convolutional layers
                        if m.stride == (1, 1) and m.padding == (1, 1):
                            w = m.weight
                            w_pos = torch.clamp(w, min=0)
                            z = torch.nn.functional.conv2d(a, w_pos, bias=None, 
                                                          stride=m.stride, padding=m.padding)
                            s = (grad_out[0] / (z + self.epsilon)).data
                            c = torch.nn.functional.conv_transpose2d(s, w_pos, 
                                                                    stride=m.stride, padding=m.padding)
                            relevances[id(m)] = (a * c).data
                        else:
                            # For stride > 1 or different padding, use a simpler rule
                            relevances[id(m)] = (a * grad_out[0]).data
                    elif isinstance(m, nn.Linear):
                        # For fully connected layers
                        w = m.weight
                        w_pos = torch.clamp(w, min=0)
                        z = torch.matmul(a, w_pos.t())
                        s = (grad_out[0] / (z + self.epsilon)).data
                        c = torch.matmul(s, w_pos)
                        relevances[id(m)] = (a * c).data
                    else:
                        # For other layer types, use a simpler propagation rule
                        relevances[id(m)] = (a * grad_out[0]).data
        
        # Register hooks for all eligible modules
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU)):
            forward_hooks.append(module.register_forward_hook(forward_hook))
            backward_hooks.append(module.register_full_backward_hook(backward_hook))
        
        # Recurse through all children
        for child in module.children():
            f_hooks, b_hooks = self._register_hooks(child, activations, relevances)
            forward_hooks.extend(f_hooks)
            backward_hooks.extend(b_hooks)
            
        return forward_hooks, backward_hooks
    
    def __call__(self, input_tensor, target_class=None):
        """
        Generates the LRP heatmap for the input tensor
        
        Args:
            input_tensor: Input image (must be normalized the same way as training data)
            target_class: Target class index. If None, uses the predicted class.
            
        Returns:
            relevance_map: The normalized LRP heatmap
        """
        input_tensor = input_tensor.clone().detach().to(device)
        input_tensor.requires_grad = True
        
        # Storage for activations and relevances
        activations = {}
        relevances = {}
        
        # Register hooks
        forward_hooks, backward_hooks = self._register_hooks(self.model, activations, relevances)
        
        try:
            # Forward pass
            output = self.model(input_tensor)
            
            # If target_class is None, use predicted class
            if target_class is None:
                target_class = torch.argmax(output, dim=1).item()
            
            # One-hot encoding for the target class
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1.0
            
            # Backward pass for LRP
            self.model.zero_grad()
            output.backward(gradient=one_hot, retain_graph=True)
            
            # Extract the input gradient as the initial relevance map
            input_gradient = input_tensor.grad.data
            
            # Get the relevance map for the first layer (closest to input)
            first_layer_id = None
            for module in self.model.modules():
                if isinstance(module, nn.Conv2d):
                    first_layer_id = id(module)
                    break
            
            if first_layer_id in relevances:
                relevance_map = relevances[first_layer_id]
            else:
                # Fallback to input gradient if we can't get the relevance map
                relevance_map = input_gradient
                
            # Sum across channels
            relevance_map = relevance_map.sum(dim=1).squeeze()
            
            # Normalize to 0-1
            relevance_map = torch.abs(relevance_map)
            if torch.max(relevance_map) > 0:
                relevance_map = relevance_map / torch.max(relevance_map)
            
            return relevance_map.cpu().numpy()
            
        finally:
            # Remove all hooks
            for hook in forward_hooks + backward_hooks:
                hook.remove()

def apply_lrp(model, img_tensor, img_np, target_class=None):
    """
    Applies LRP to visualize model contributions
    
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
    
    # Generate relevance map
    relevance_map = lrp(img_tensor, target_class)
    
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

def visualize_lrp(model, dataloader, class_names, num_images=5):
    """
    Visualizes LRP for a batch of images
    
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
        img_np = np.clip(img_np * np.array([0.229, 0.224, 0.225]) + 
                        np.array([0.485, 0.456, 0.406]), 0, 1)
        
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
        axes[i, 0].set_title(f"True: {class_names[label]}\nPred: {class_names[pred]} ({prob[0][pred.item()]:.2f})")
        axes[i, 0].axis('off')
        
        # Display LRP for true class
        axes[i, 1].imshow(true_lrp)
        axes[i, 1].set_title(f"LRP for {class_names[label]}")
        axes[i, 1].axis('off')
        
        # Display LRP for predicted class
        axes[i, 2].imshow(pred_lrp)
        axes[i, 2].set_title(f"LRP for {class_names[pred]}")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('lrp_visualizations.png')
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
        img_np = np.clip(img_np * np.array([0.229, 0.224, 0.225]) + 
                        np.array([0.485, 0.456, 0.406]), 0, 1)
        
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
        
        # Display original image
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title(f"Original\nTrue: {class_names[label]}\nPred: {class_names[pred]} ({prob[0][pred.item()]:.2f})")
        axes[i, 0].axis('off')
        
        # Display GradCAM
        axes[i, 1].imshow(gradcam_vis)
        axes[i, 1].set_title("GradCAM")
        axes[i, 1].axis('off')
        
        # Display LRP
        axes[i, 2].imshow(lrp_vis)
        axes[i, 2].set_title("Layer-wise Relevance Propagation")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('xai_comparison.png')
    plt.show()
    
    # Add a detailed analysis
    print("\nXAI Methods Comparison Analysis:")
    print("-------------------------------")
    print("GradCAM:")
    print("  - Highlights regions that most strongly activate the target class.")
    print("  - Focuses on the last convolutional layer, which may not capture fine details.")
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
    print("  - For complex features (like dog breeds), these visualizations help identify")
    print("    which visual traits the model is using to distinguish between classes.")

    