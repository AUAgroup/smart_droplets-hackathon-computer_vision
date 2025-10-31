import warnings
import os
import csv
import math
import random
import copy

import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import lr_scheduler

# Third-Party Libraries (Computer Vision/ML)
import timm # PyTorch Image Models
import cv2 # OpenCV
# import albumentations as A # Uncomment if needed for augmentations
# import pytorch_lightning as pl # Uncomment if using PyTorch Lightning

# PyTorch Vision
import torchvision
from torchvision import transforms
from torchvision import transforms as T # Alias kept for flexibility
from torchvision.transforms import functional as TF
from torchvision.transforms import v2 # Assuming v2 is necessary for specific transforms

# External/Project Utilities
from PIL import Image
from torchmetrics.classification import MulticlassJaccardIndex # type: ignore

import re