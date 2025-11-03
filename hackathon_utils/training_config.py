from hackathon_utils.data_config import DATASET_NAME

# ---------------------------------------------------------------------
# Global Training Configuration
# ---------------------------------------------------------------------

# Source of pretrained weights for the model backbone.
# Common options: "IMAGENET", "RANDOM", "SSL" (self-supervised)
WEIGHTS_INITIALIZATION = "IMAGENET"

# Directory where all visualizations, logs, and checkpoints will be saved.
# Uses dataset and initialization to keep experiments organized.
OUTPUT_FOLDER_NAME = f"./segmentation_evaluation-{DATASET_NAME}-{WEIGHTS_INITIALIZATION}"

# ---------------------------------------------------------------------
# Core Training Hyperparameters
# ---------------------------------------------------------------------

# Total number of training epochs (complete passes through the dataset).
NUM_EPOCHS = 10

# Batch size for both training and validation dataloaders.
# Determines how many samples are processed before gradient update.
BATCH_SIZE = 8

# Early stopping patience: number of epochs to wait without improvement 
# before halting training and restoring the best model weights.
PATIENCE = 10

# ---------------------------------------------------------------------
# Model Architecture and Optimization Parameters
# ---------------------------------------------------------------------

# Segmentation architecture used (supported by segmentation_models_pytorch or similar).
# Examples: "unet", "unetplusplus", "deeplabv3", "segformer", etc.
ARCHITECTURE_NAME = "unetplusplus"

# Backbone network used for feature extraction (defined by timm library name).
# "maxvit_tiny_tf_512.in1k" = MaxViT Tiny pretrained on ImageNet-1k at 512x512 input size.
BACKBONE_NAME = "maxvit_tiny_tf_512.in1k"

# Weight decay (L2 regularization) used to prevent overfitting.
WEIGHT_DECAY = 1e-4

# Base learning rate for optimizer (typically AdamW).
# Controls step size in gradient updates.
LEARNING_RATE = 5e-4
