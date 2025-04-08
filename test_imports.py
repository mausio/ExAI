#!/usr/bin/env python3
# Simple script to test if all required packages are installed correctly

# Basic imports
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from PIL import Image

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.datasets.utils import download_url, extract_archive
import torchvision

# For evaluation metrics
from sklearn.metrics import confusion_matrix, classification_report

print("All imports successful!")
print(f"PyTorch version: {torch.__version__}")
print(f"TorchVision version: {torchvision.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Using CUDA: {torch.cuda.is_available()}")

if __name__ == "__main__":
    print("\nYou're all set to run the corgi classifier!") 