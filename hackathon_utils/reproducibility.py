import torch
import numpy as np
import random
import os

def set_seed(seed: int = 2025):
    """
    Set all relevant random seeds for full reproducibility across Python, NumPy, and PyTorch.

    This function ensures deterministic behavior (as much as possible) across:
      - Python’s built-in random module
      - NumPy random generator
      - PyTorch (CPU + CUDA)
      - cuDNN backend settings (to avoid non-deterministic kernels)
      - Multi-GPU or multi-worker dataloaders (when using torch DataLoader)

    Args:
        seed (int): Random seed value. Default = 2025.
    """

    # --- 1. Python's built-in random module ---
    random.seed(seed)

    # --- 2. NumPy random generator ---
    np.random.seed(seed)

    # --- 3. PyTorch (CPU) ---
    torch.manual_seed(seed)

    # --- 4. PyTorch (CUDA) ---
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Ensures all GPUs use the same seed

    # --- 5. Enforce deterministic behavior in cuDNN ---
    # cudnn.benchmark = False: ensures deterministic results instead of autotuned kernels
    # cudnn.deterministic = True: disables nondeterministic algorithms
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # --- 6. Control Python’s hashing (useful in distributed settings or dataloaders) ---
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Optional: log confirmation (if logger is available)
    print(f"[Seed set] Reproducibility enforced with seed = {seed}")