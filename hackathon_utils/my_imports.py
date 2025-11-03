# ---------------------------------------------------------------------
# ⚠️ General Imports and Environment Setup
# ---------------------------------------------------------------------
# These imports cover system utilities, debugging, numerical processing,
# visualization, and core ML/vision libraries used throughout the project.
# The structure is grouped by purpose: system → scientific → ML → project-specific.

# -------------------------------
# Core Python / System Libraries
# -------------------------------
import warnings        # For managing runtime warnings (e.g., deprecation, numerical issues)
import os              # Filesystem path and directory operations
import csv             # Reading/writing structured data (e.g., metrics logs)
import math            # Mathematical operations (log, sqrt, etc.)
import random          # Random sampling, augmentations, reproducibility
import copy            # Deepcopying model weights or configurations
import traceback       # For detailed error diagnostics during training/evaluation

from collections import defaultdict  # Simplified dictionary management with default values
from pathlib import Path             # Object-oriented filesystem paths (safer than os.path)
import re                            # Regular expressions (for filename or pattern matching)

# -------------------------------
# Numerical / Visualization Stack
# -------------------------------
import numpy as np                   # Core numerical library for array processing
import matplotlib.pyplot as plt      # Visualization backend (plots, learning curves)
from matplotlib.colors import ListedColormap  # For color mapping segmentation masks

# -------------------------------
# PyTorch Core Framework
# -------------------------------
import torch                         # Main PyTorch package
import torch.nn as nn                # Neural network building blocks
import torch.nn.functional as F      # Functional API for activations/losses
from torch.utils.data import DataLoader, Dataset, random_split  # Dataset utilities
from torch.optim import lr_scheduler  # Learning rate schedulers

# -------------------------------
# Third-Party CV / ML Libraries
# -------------------------------
import timm   # PyTorch Image Models — for pre-trained backbones and model creation
import cv2    # OpenCV — used for image loading, augmentation, or preprocessing

# Optional integrations (commented out for flexibility):
# import albumentations as A     # Advanced data augmentations (uncomment if needed)
# import pytorch_lightning as pl # Framework for structured training loops

# -------------------------------
# TorchVision: Image Transformations & Utilities
# -------------------------------
import torchvision
from torchvision import transforms                # Common image transformations (resize, normalize, etc.)
from torchvision import transforms as T           # Alias maintained for backward compatibility
from torchvision.transforms import functional as TF  # Functional image ops (rotate, crop, flip)
from torchvision.transforms import v2             # v2 API for newer torchvision augmentations

# -------------------------------
# External / Project-Specific Utilities
# -------------------------------
from PIL import Image                             # Pillow for image I/O and format conversions
from torchmetrics.classification import MulticlassJaccardIndex  # IoU (Jaccard) metric for segmentation
