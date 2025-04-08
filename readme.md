# ExAI - Explainable Corgi (Cardigan) Separator 🐶

This project uses explainable AI (XAI) techniques to visualize and understand how a deep learning model distinguishes between two breeds of Welsh Corgis: Pembroke and Cardigan.

## Overview

The goal of this project is to train a CNN (ResNet50) to classify Pembroke vs Cardigan Welsh Corgis and then apply XAI methods to:
1. Understand what features the model is focusing on when making decisions
2. Examine how the model behaves with mixed-breed or ambiguous examples
3. Compare different XAI approaches for model interpretation

## XAI Methods Implemented

### 1. Contrastive Grad-CAM
This method extends the traditional Grad-CAM approach by explicitly comparing activation maps between two classes. This highlights regions that are discriminative for each class, making it easier to understand what features the model uses to distinguish between Pembroke and Cardigan Corgis.

### 2. Layerwise Relevance Propagation (LRP)
LRP provides a pixel-level explanation by backpropagating the model's decision to the input pixels. This gives a more fine-grained visualization of which image regions contribute positively or negatively to the classification.

## Dataset

We use the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/), specifically the classes for Welsh Corgi Pembroke and Welsh Corgi Cardigan.

## Project Structure

- `main.ipynb`: Jupyter notebook containing all steps from data loading to model training and XAI analysis
- `gradcam_methods/`: Python module containing XAI implementations
  - `gradcam.py`: Implementation of the Contrastive Grad-CAM method
  - `lrp.py`: Implementation of the Layerwise Relevance Propagation method
- `demo_xai.py`: Demonstration script for applying XAI methods to test images
- `presentation/`: Contains presentation slides

## Usage

```sh
pip install numpy matplotlib seaborn tqdm Pillow opencv-python torch torchvision scikit-learn
```

### Setting Up

1. Install dependencies:
```
pip install torch torchvision matplotlib numpy pandas tqdm opencv-python
```

2. Download the dataset:
```
# The notebook contains download code using the Stanford Dogs Dataset
```

### Training the Model

Run the `main.ipynb` notebook to:
- Prepare and load the dataset
- Fine-tune a ResNet50 model on Corgi classification
- Apply XAI methods to understand model predictions

### Using the Demo Script

To apply XAI methods to a single image:

```
python demo_xai.py --img_path /path/to/corgi/image.jpg --method both
```

Options:
- `--img_path`: Path to the test image
- `--method`: XAI method to use ('gradcam', 'lrp', or 'both')
- `--model_path`: Path to the finetuned model (default: 'resnet50_finetuned.pth')
- `--output_dir`: Directory to save results (default: 'results')

## Example Results

![Contrastive Grad-CAM Example](presentation/images/gradcam_example.png)
*Contrastive Grad-CAM highlighting features the model uses to distinguish Pembroke vs Cardigan*

![LRP Example](presentation/images/lrp_example.png)
*LRP visualization showing pixel-level contributions to the model's decision*

## References

- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)
- [Layer-Wise Relevance Propagation for Neural Networks](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)
- [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [ResNet](https://arxiv.org/abs/1512.03385)

## Contributors

- Lukas
- Janik  
- Robin
- Felix 