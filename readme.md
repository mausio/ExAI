# ExAI - Explainable Corgi (Cardigan) Separator 🐶

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project uses explainable AI (XAI) techniques to visualize and understand how a deep learning model distinguishes between two breeds of Welsh Corgis: Pembroke and Cardigan.

![Corgi Comparison](https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Pembroke_and_Cardigan_Welsh_Corgis.jpg/800px-Pembroke_and_Cardigan_Welsh_Corgis.jpg)
*Left: Cardigan Welsh Corgi with a tail. Right: Pembroke Welsh Corgi with docked tail.*

## Table of Contents
- [Overview](#overview)
- [Corgi Breed Differences](#corgi-breed-differences)
- [XAI Methods Implemented](#xai-methods-implemented)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results and Visualization](#results-and-visualization)
- [Challenges and Future Work](#challenges-and-future-work)
- [References](#references)
- [Contributors](#contributors)

## Overview

The goal of this project is to train a CNN (ResNet50) to classify Pembroke vs Cardigan Welsh Corgis and then apply XAI methods to:
1. Understand what features the model is focusing on when making decisions
2. Examine how the model behaves with mixed-breed or ambiguous examples
3. Compare different XAI approaches for model interpretation
4. Verify if the model focuses on the same anatomical differences that experts use to distinguish these breeds

## Corgi Breed Differences

Despite being closely related, Pembroke and Cardigan Welsh Corgis have several distinguishing features:

1. **Tail**: Cardigans have long, fox-like tails while Pembrokes typically have docked tails or are born with naturally bobbed tails
2. **Ears**: Cardigans have larger, more rounded ears
3. **Body**: Cardigans are generally longer-bodied with heavier bone structure
4. **Feet**: Cardigans have round feet, while Pembrokes have more oval-shaped feet
5. **Coat Colors**: Cardigans come in more color variations, including brindle and blue merle patterns

These subtle differences make this an interesting classification problem even for humans. Our XAI approaches aim to determine if a neural network focuses on these same distinctive features.

## XAI Methods Implemented

### 1. Contrastive Grad-CAM
This method extends the traditional Grad-CAM approach by explicitly comparing activation maps between two classes. This highlights regions that are discriminative for each class, making it easier to understand what features the model uses to distinguish between Pembroke and Cardigan Corgis.

Key features:
- Targets specific layers of the CNN (typically the final convolutional layer)
- Produces coarse localization maps highlighting important regions
- Allows direct comparison between class-specific activation patterns

### 2. Layerwise Relevance Propagation (LRP)
LRP provides a pixel-level explanation by backpropagating the model's decision to the input pixels. This gives a more fine-grained visualization of which image regions contribute positively or negatively to the classification.

Key features:
- Provides higher resolution explanations than Grad-CAM
- Shows both positive and negative contributions to the decision
- Creates more detailed heatmaps showing exact pixel relevance
- Follows a conservation principle where relevance is neither created nor destroyed

## Model Architecture

We use a transfer learning approach with ResNet50:

- **Base Network**: ResNet50 pre-trained on ImageNet
- **Transfer Learning Strategy**: 
  - Freeze early layers (capturing generic features)
  - Fine-tune only Layer4 and classification head (for domain-specific features)
- **Classification Head**:
  - Custom fully-connected layer (2048 → 512 → 2)
  - ReLU activation and dropout (0.3) for regularization
- **Training Parameters**:
  - Loss Function: Cross-Entropy Loss
  - Optimizer: Adam with different learning rates for Layer4 (1e-4) and FC layer (1e-3)
  - Learning Rate Scheduler: ReduceLROnPlateau with patience=3
  - Early Stopping: 5 epochs without improvement

## Dataset

We use the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/), specifically the classes for Welsh Corgi Pembroke and Welsh Corgi Cardigan. The dataset is automatically downloaded and extracted in the notebook.

- **Split**: 80% training, 20% validation
- **Data Augmentation**: Random horizontal flips, slight rotations, and color jitter
- **Preprocessing**: Resize to 224x224, normalize with ImageNet statistics

## Project Structure

- `main.ipynb`: Jupyter notebook containing all steps from data loading to model training and XAI analysis
- `gradcam_methods/`: Python module containing XAI implementations
  - `gradcam.py`: Implementation of the Contrastive Grad-CAM method
  - `lrp.py`: Implementation of the Layerwise Relevance Propagation method
- `demo_xai.py`: Demonstration script for applying XAI methods to test images
- `presentation/`: Contains presentation slides

## Installation

### Requirements

```
python>=3.7
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.4.0
numpy>=1.19.5
tqdm>=4.62.0
opencv-python>=4.5.3
scikit-learn>=0.24.2
seaborn>=0.11.2
pillow>=8.3.1
```

```sh
pip install numpy matplotlib seaborn tqdm Pillow opencv-python torch torchvision scikit-learn
```

### Setting Up

1. Clone the repository:
```bash
git clone https://github.com/yourusername/exai-corgi-separator.git
cd exai-corgi-separator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
```
# The notebook contains download code using the Stanford Dogs Dataset
```

## Usage

### Training the Model

Run the `main.ipynb` notebook to:
- Prepare and load the dataset
- Fine-tune a ResNet50 model on Corgi classification
- Apply XAI methods to understand model predictions

### Using the Demo Script

To apply XAI methods to a single image:

```bash
python demo_xai.py --img_path /path/to/corgi/image.jpg --method both
```

Options:
- `--img_path`: Path to the test image
- `--method`: XAI method to use ('gradcam', 'lrp', or 'both')
- `--model_path`: Path to the finetuned model (default: 'resnet50_finetuned.pth')
- `--output_dir`: Directory to save results (default: 'results')

## Results and Visualization

Our model achieves over 90% accuracy in distinguishing between the two Corgi breeds. The XAI methods reveal interesting insights:

### GradCAM Observations
- For Cardigan Welsh Corgis, the model often focuses on the tail region and the somewhat larger ears
- For Pembroke Welsh Corgis, the model focuses on the back region where tails would be absent and the more oval body shape

### LRP Observations
- LRP provides more granular explanations, highlighting specific pixels that contribute to the decision
- LRP shows that the model also considers coat color patterns to some extent, particularly in the brindle patterns sometimes seen in Cardigans

![Combined Visualization](presentation/images/xai_comparison.png)
*Comparison of original image, GradCAM, and LRP visualizations for the same Corgi image*

## Challenges and Future Work

- **Data Limitations**: The Stanford Dogs Dataset contains a limited number of Corgi images, which can impact model generalization
- **Mixed Breeds**: Testing with mixed Pembroke-Cardigan Corgis would provide interesting edge cases
- **Alternative XAI Methods**: Implementing additional explanation methods like LIME or SHAP could provide further insights
- **User Study**: Conducting a study with dog experts to compare human vs. AI reasoning would validate our approach
- **Model Distillation**: Training a smaller, more interpretable model that mimics the behavior of ResNet50

## References

- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- [Layer-Wise Relevance Propagation for Neural Networks](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)
- [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [ResNet: Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [American Kennel Club - Cardigan vs. Pembroke Welsh Corgis](https://www.akc.org/expert-advice/lifestyle/cardigan-vs-pembroke-welsh-corgi/)

## Contributors

- Lukas
- Janik  
- Robin
- Felix 