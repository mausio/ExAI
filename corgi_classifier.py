###############################################################################
# start with ".venv/bin/python corgi_classifier.py"

# Path to the specific image
demo_mischling = "./assets/corgi-mischling.jpeg"
demo_pembroke = "./assets/corgi-pembroke.jpeg"
###############################################################################

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from torchvision.datasets.utils import download_url, extract_archive

from sklearn.metrics import confusion_matrix, classification_report

from xai_methods import visualize_gradcam, visualize_lrp, compare_xai_methods, compare_gradcam_classes, compare_lrp_classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def download_and_extract_dataset(download_dir, extract_dir):
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    
    dataset_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    filename = os.path.basename(dataset_url)
    filepath = os.path.join(download_dir, filename)
    
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        download_url(dataset_url, download_dir)
    else:
        print(f"File {filename} already exists in {download_dir}")
    
    if not os.path.exists(os.path.join(extract_dir, "Images")):
        print(f"Extracting {filename} to {extract_dir}...")
        extract_archive(filepath, extract_dir)
    else:
        print(f"Dataset already extracted to {extract_dir}")

class CorgiDataset(Dataset):
    def __init__(self, dataset_root, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = ['Pembroke', 'Cardigan']
        
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
        
        for img_name in os.listdir(pembroke_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(pembroke_dir, img_name))
                self.labels.append(0)
        
        for img_name in os.listdir(cardigan_dir):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                self.image_paths.append(os.path.join(cardigan_dir, img_name))
                self.labels.append(1)
        
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
            blank_image = torch.zeros((3, 224, 224)) if self.transform else Image.new('RGB', (224, 224), (0, 0, 0))
            return blank_image, self.labels[idx]

class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, idx):
        image, label = self.subset.dataset[self.subset.indices[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.subset)

def prepare_dataloaders(dataset_root, batch_size=32, num_workers=2):
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
    
    dataset = CorgiDataset(dataset_root, transform=None)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataset_transformed = TransformedSubset(train_dataset, data_transforms['train'])
    val_dataset_transformed = TransformedSubset(val_dataset, data_transforms['val'])
    
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

def setup_model(num_classes=2):
    model = models.resnet50(weights='DEFAULT')
    
    for param in model.parameters():
        param.requires_grad = False
    
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    
    model = model.to(device)
    
    print(f"ResNet50 model configured for {num_classes} classes")
    parameters_of_layer4 = sum(p.numel() for p in model.layer4.parameters() if p.requires_grad)
    print(f"Trainable parameters in output layer 'model.layer4': {parameters_of_layer4}")
    parameters_of_fully_connected_layers = sum(p.numel() for p in model.fc.parameters() if p.requires_grad)
    print(f"Trainable parameters in fully connected layers 'model.fc': {parameters_of_fully_connected_layers}")
    
    return model

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
    return epoch_loss, epoch_acc.item()

def validate_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    
    print(f'Validation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    
    return epoch_loss, epoch_acc.item()

def train_model(model, train_loader, val_loader, num_epochs=5, patience=5):
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-5)
    
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
        )
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    no_improve_epochs = 0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 40)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_loss)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
        
        print(f'Best val Acc: {best_acc:.4f}')
        
        if no_improve_epochs >= patience:
            print(f'Early stopping after {epoch+1} epochs without improvement')
            break
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    
    model.load_state_dict(best_model_wts)
    return model, history

def evaluate_model(model, dataloader, class_names):
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
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()
    
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    return y_true, y_pred, report

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
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

def save_model(model, save_path, class_names, optimizer=None, epoch=None, history=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(model, save_path.replace('.pth', '_full.pth'))
    print(f"Complete model saved to: {save_path.replace('.pth', '_full.pth')}")
    
    torch.save(model.state_dict(), save_path.replace('.pth', '_weights.pth'))
    print(f"Model weights saved to: {save_path.replace('.pth', '_weights.pth')}")
    
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
    try:
        checkpoint = torch.load(load_path, map_location=device)
        
        if model is None:
            if load_path.endswith('_full.pth'):
                model = torch.load(load_path, map_location=device)
                print(f"Full model loaded from: {load_path}")
                return model, None
            
            model = setup_model()
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"Model weights loaded from: {load_path}")
        return model, checkpoint
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
class SingleImageDataset(Dataset):
    def __init__(self, image_path, transform=None):
        """
        Dataset for loading a single image without a label
        
        Args:
            image_path: Path to the image file
            transform: PyTorch transforms for preprocessing
        """
        self.image_path = image_path
        self.transform = transform
        
    def __len__(self):
        return 1  # Only one image
        
    def __getitem__(self, idx):
        # Load image
        try:
            image = Image.open(self.image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            return image, -1  # -1 as a placeholder label since we don't have one
        except Exception as e:
            print(f"Error loading image {self.image_path}: {e}")
            # Return a blank image
            blank_image = torch.zeros((3, 224, 224)) if self.transform else Image.new('RGB', (224, 224), (0, 0, 0))
            return blank_image, -1

def apply_xai_to_trained_model(model, val_loader, class_names, num_images=5):
    """Apply different XAI methods to the trained model for interpretability"""
    print("\n" + "="*50)
    print("Applying XAI Methods for Model Interpretability")
    print("="*50)

    print("\nGenerating GradCAM visualizations...")
    visualize_gradcam(model, val_loader, class_names, num_images=num_images)

    print("\nGenerating Layer-wise Relevance Propagation visualizations...")
    visualize_lrp(model, val_loader, class_names, num_images=num_images)

    print("\nComparing GradCAM and LRP methods...")
    compare_xai_methods(model, val_loader, class_names, num_images=3)

    print("\nXAI visualization complete. All results saved as PNG files.")

def analyze_demo_images(model, class_names):
    print("\nLoading manual images for inference...")
    manual_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # First analyze the Pembroke image
    print(f"\nAnalyzing Pembroke demo image: {demo_pembroke}")
    pembroke_dataset = SingleImageDataset(demo_pembroke, transform=manual_transform)
    pembroke_loader = DataLoader(pembroke_dataset, batch_size=1, shuffle=False)

    print("Applying XAI methods to Pembroke image:")
    compare_xai_methods(model, pembroke_loader, class_names, num_images=1)
    
    print("Comparing GradCAM for different classes on Pembroke image:")
    compare_gradcam_classes(model, pembroke_loader, class_names, num_images=1)
    
    print("Comparing LRP for different classes on Pembroke image:")
    compare_lrp_classes(model, pembroke_loader, class_names, num_images=1)

    # Then analyze the mixed-breed image
    print(f"\nAnalyzing mixed-breed demo image: {demo_mischling}")
    mischling_dataset = SingleImageDataset(demo_mischling, transform=manual_transform)
    mischling_loader = DataLoader(mischling_dataset, batch_size=1, shuffle=False)

    print("Applying XAI methods to mixed-breed image:")
    compare_xai_methods(model, mischling_loader, class_names, num_images=1)
    
    print("Comparing GradCAM for different classes on mixed-breed image:")
    compare_gradcam_classes(model, mischling_loader, class_names, num_images=1)
    
    print("Comparing LRP for different classes on mixed-breed image:")
    compare_lrp_classes(model, mischling_loader, class_names, num_images=1)

def main():
    download_dir = "./downloads"
    extract_dir = "./dogs"
    number_of_epochs = 3
    
    images_to_apply_xai_on = 5
    
    download_and_extract_dataset(download_dir, extract_dir)
    
    train_loader, val_loader, class_names = prepare_dataloaders(extract_dir, batch_size=32)
    
    model = setup_model(num_classes=len(class_names))
    
    os.makedirs(download_dir, exist_ok=True)
    model_path = os.path.join(download_dir, 'resnet50_corgi_classifier.pth')
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        model, checkpoint = load_model(model_path)
    else:
        print("Training a new model...")
        model, history = train_model(model, train_loader, val_loader, number_of_epochs)
        
        plot_training_history(history)
        
        y_true, y_pred, report = evaluate_model(model, val_loader, class_names)
        
        save_model(
            model, 
            model_path,
            class_names=class_names,
            history=history
        )

        # Apply XAI methods to the trained model
        apply_xai_to_trained_model(model, val_loader, class_names, num_images=images_to_apply_xai_on)

    # Analyze demo images using XAI methods
    analyze_demo_images(model, class_names)

if __name__ == "__main__":
    main()