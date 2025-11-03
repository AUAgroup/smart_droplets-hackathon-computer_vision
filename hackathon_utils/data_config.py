# ---------------------------------------------------------------------
# Dataset Configuration
# ---------------------------------------------------------------------

# Dataset root name (used for building paths automatically)
# Each subfolder under this dataset contains train/val/test splits
DATASET_NAME = "smart_droplets-scab_hackathon-2_classes-checked-patched_512-splits"

# ---------------------------------------------------------------------
# Folder Paths
# ---------------------------------------------------------------------

# Training set: images and corresponding segmentation masks
TRAIN_IMG_DIR = f'./{DATASET_NAME}/train/images'
TRAIN_MASK_DIR = f'./{DATASET_NAME}/train/masks'

# Validation set: used to monitor generalization and apply early stopping
VAL_IMG_DIR = f'./{DATASET_NAME}/val/images'
VAL_MASK_DIR = f'./{DATASET_NAME}/val/masks'

# Test set: unseen data used for final model evaluation
TEST_IMG_DIR = f'./{DATASET_NAME}/test/images'

# Folder to save predicted masks during inference/testing
RESULTS_IMG_DIR = f'./{DATASET_NAME}/test/pred_masks'

# ---------------------------------------------------------------------
# Image and Model Parameters
# ---------------------------------------------------------------------

# Expected image input size (used for resizing and model config)
IMAGE_SIZE = 512

# Patch size (if dataset was patch-extracted, same as image size for simplicity)
PATCH_SIZE = 512

# Number of segmentation classes (e.g., 2 for binary: background vs. target)
NUM_CLASSES = 2

# Label index of the main or problematic class to monitor more closely (e.g., disease)
PROBLEMATIC_CLASS = 1

# Whether the dataset includes a validation split (some datasets only have train/test)
INCLUDES_VAL = True
