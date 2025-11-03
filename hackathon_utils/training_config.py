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
# Try a larger number for potentially better results, but training time increases.
NUM_EPOCHS = 10

# Batch size for both training and validation dataloaders.
# Determines how many samples are processed before gradient update.
# Try to use the largest batch size that fits in GPU memory for (maybe) better results.
BATCH_SIZE = 4

# Early stopping patience: number of epochs to wait without improvement 
# before halting training and restoring the best model weights.
PATIENCE = 10

# ---------------------------------------------------------------------
# Model Architecture and Optimization Parameters
# ---------------------------------------------------------------------

# Segmentation architecture used (supported by segmentation_models_pytorch or similar).
# Examples: "unet", "unetplusplus", "deeplabv3", "segformer", etc.
# More: https://smp.readthedocs.io/en/latest/models.html
ARCHITECTURE_NAME = "unet"

# Backbone network used for feature extraction (defined by timm library name).
# "maxvit_tiny_tf_512.in1k" = MaxViT Tiny pretrained on ImageNet-1k at 512x512 input size.
# Examples: "convnextv2_tiny.fcmae_ft_in22k_in1k_384", "resnet34.a1_in1k", "maxvit_tiny_tf_512.in1k"
# More: https://smp.readthedocs.io/en/latest/encoders.html#
# More: https://smp.readthedocs.io/en/latest/encoders_timm.html
BACKBONE_NAME = "convnextv2_tiny.fcmae_ft_in22k_in1k_384"

# Weight decay (L2 regularization) used to prevent overfitting.
WEIGHT_DECAY = 1e-4

# Base learning rate for optimizer (typically AdamW).
# Controls step size in gradient updates.
LEARNING_RATE = 5e-4
