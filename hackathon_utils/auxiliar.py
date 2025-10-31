import os
import torch
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable

def mean_values_by_key(list_of_dicts):
    """
    Calculates the mean of float values for each key across a list of dictionaries.

    Args:
        list_of_dicts: A list of dictionaries where keys are numbers and values are floats.

    Returns:
        A dictionary with keys from the original dictionaries and values as the
        mean of the corresponding float values.
    """
    sum_values = defaultdict(float)
    count_values = defaultdict(int)

    for d in list_of_dicts:
        for key, value in d.items():
            if isinstance(key, (int, float)) and isinstance(value, float):
                sum_values[key] += value
                count_values[key] += 1
            else:
                print(f"Warning: Skipping non-numeric key or non-float value: Key '{key}' (type: {type(key).__name__}), Value '{value}' (type: {type(value).__name__})")

    final_means = {}
    for key in sum_values:
        if count_values[key] > 0:
            final_means[key] = round(sum_values[key] / count_values[key], 4)
        else:
            final_means[key] = 0.0 # Or handle as appropriate for keys with no values

    return final_means

def create_folder_if_not_exists(folder_path):
    """
    Checks if a folder exists at the given path. If it does not exist,
    the folder is created.

    Args:
        folder_path (str): The path to the folder to check/create.
    """
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            logger.info(f"Folder '{folder_path}' created successfully.")
        except OSError as e:
            logger.error(f"Error creating folder '{folder_path}': {e}")
    else:
        logger.info(f"Folder '{folder_path}' already exists.")