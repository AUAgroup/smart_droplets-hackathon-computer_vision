import os
import torch
import logging
from collections import defaultdict

# ---------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------

# Create a logger specific to this module.
# Using NullHandler prevents warnings if no global logging configuration exists.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ---------------------------------------------------------------------
# Function: count_params
# ---------------------------------------------------------------------
def count_params(m):
    """
    Counts the total and trainable parameters of a PyTorch model.

    Args:
        m (torch.nn.Module): Model instance.

    Returns:
        (total_params, trainable_params): Tuple of integers.
            - total_params: total number of parameters in the model.
            - trainable_params: subset of parameters with requires_grad=True.
    """
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

# ---------------------------------------------------------------------
# Function: mean_values_by_key
# ---------------------------------------------------------------------
def mean_values_by_key(list_of_dicts):
    """
    Computes the mean of float values for each key across a list of dictionaries.

    Commonly used to average per-class metrics (e.g., IoU per class)
    over multiple batches or evaluation steps.

    Args:
        list_of_dicts (list[dict]):
            A list where each element is a dictionary with:
                - numeric keys (e.g., class indices)
                - float values (e.g., IoU scores)

    Returns:
        dict:
            Dictionary mapping each key to the mean of its float values
            across all dictionaries in the list.
    """
    sum_values = defaultdict(float)
    count_values = defaultdict(int)

    for d in list_of_dicts:
        for key, value in d.items():
            # Only consider numeric keys and float values
            if isinstance(key, (int, float)) and isinstance(value, float):
                sum_values[key] += value
                count_values[key] += 1
            else:
                logger.info(
                    f"Warning: Skipping non-numeric key or non-float value: "
                    f"Key '{key}' (type: {type(key).__name__}), "
                    f"Value '{value}' (type: {type(value).__name__})"
                )

    # Compute means safely (avoid division by zero)
    final_means = {}
    for key in sum_values:
        if count_values[key] > 0:
            final_means[key] = round(sum_values[key] / count_values[key], 4)
        else:
            final_means[key] = 0.0  # fallback for keys with no valid values

    return final_means

# ---------------------------------------------------------------------
# Function: create_folder_if_not_exists
# ---------------------------------------------------------------------
def create_folder_if_not_exists(folder_path):
    """
    Ensures a directory exists at the given path.
    Creates it if missing, and logs the result.

    Args:
        folder_path (str): Path to the target folder.
    """
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            logger.info(f"Folder '{folder_path}' created successfully.")
        except OSError as e:
            logger.error(f"Error creating folder '{folder_path}': {e}")
    else:
        logger.info(f"Folder '{folder_path}' already exists.")
