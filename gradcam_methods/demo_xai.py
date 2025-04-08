import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse

# Import our XAI methods
from gradcam_methods.gradcam import GradCAM, generate_contrastive_gradcam, visualize_contrastive_gradcam
from gradcam_methods.lrp import LRP, visualize_lrp, compare_methods

def parse_args():
    parser = argparse.ArgumentParser(description='XAI methods for Corgi classification')
    parser.add_argument('--model_path', type=str, default='resnet50_finetuned.pth', help='Path to finetuned model')
    parser.add_argument('--img_path', type=str, required=True, help='Path to test image')
    parser.add_argument('--method', type=str, default='both', choices=['gradcam', 'lrp', 'both'], 
                        help='XAI method to use')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    return parser.parse_args()

def load_model(model_path):
    # Load pre-trained ResNet50
    model = models.resnet50(pretrained=False)
    
    # Modify for our 2-class problem
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: Pembroke and Cardigan
    
    # Load fine-tuned weights
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found. Using pretrained model.")
    
    model.eval()
    return model

def prepare_image(img_path):
    # Image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)
    
    return img_tensor

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path)
    
    # Prepare image
    img_tensor = prepare_image(args.img_path)
    
    # Get class predictions
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]
        pembroke_prob = probs[0].item()  # Assuming class 0 is Pembroke
        cardigan_prob = probs[1].item()  # Assuming class 1 is Cardigan
        predicted_class = outputs.argmax(dim=1).item()
        class_name = "Pembroke" if predicted_class == 0 else "Cardigan"
    
    print(f"Prediction: {class_name} (Pembroke: {pembroke_prob:.2%}, Cardigan: {cardigan_prob:.2%})")
    
    # Apply XAI methods based on argument
    if args.method == 'gradcam' or args.method == 'both':
        # Apply Contrastive Grad-CAM
        print("Applying Contrastive Grad-CAM...")
        cam_pembroke, cam_cardigan, diff_cam = generate_contrastive_gradcam(
            model, img_tensor, class_idx_a=0, class_idx_b=1
        )
        
        # Visualize and save
        plt.figure(figsize=(15, 10))
        visualize_contrastive_gradcam(
            img_tensor, cam_pembroke, cam_cardigan, diff_cam,
            class_a_name="Pembroke", class_b_name="Cardigan"
        )
        plt.savefig(os.path.join(args.output_dir, 'gradcam_results.png'))
        print(f"Grad-CAM results saved to {os.path.join(args.output_dir, 'gradcam_results.png')}")
    
    if args.method == 'lrp' or args.method == 'both':
        # Apply LRP
        print("Applying Layerwise Relevance Propagation...")
        lrp = LRP(model)
        lrp_heatmap = lrp.generate(img_tensor, predicted_class)
        
        # Visualize and save
        plt.figure(figsize=(12, 5))
        visualize_lrp(img_tensor, lrp_heatmap, class_name=class_name)
        plt.savefig(os.path.join(args.output_dir, 'lrp_results.png'))
        print(f"LRP results saved to {os.path.join(args.output_dir, 'lrp_results.png')}")
    
    if args.method == 'both':
        # Compare methods
        print("Comparing both XAI methods...")
        compare_results = compare_methods(model, args.img_path)
        
        plt.figure(figsize=(15, 10))
        plt.savefig(os.path.join(args.output_dir, 'comparison_results.png'))
        print(f"Comparison results saved to {os.path.join(args.output_dir, 'comparison_results.png')}")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 