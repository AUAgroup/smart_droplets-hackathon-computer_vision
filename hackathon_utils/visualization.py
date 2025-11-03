import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from hackathon_utils.training_config import *

# ---------------------------------------------------------------------
# Utility function: collect_two_class_samples
# ---------------------------------------------------------------------
def collect_two_class_samples(
    dataloader,
    device,
    n_needed: int = 6,
    min_unique_classes: int = 2,
    max_passes: int = 3,  # maximum number of dataloader iterations before stopping
):
    """
    Collect a specified number of samples (images, masks) where each mask 
    contains at least `min_unique_classes` unique class labels.
    Useful for visualization of segmentation models (ensures both classes present).

    Args:
        dataloader: PyTorch dataloader yielding (images [B, C, H, W], masks [B, H, W] or [B,1,H,W])
        device: torch.device or string ('cuda' or 'cpu')
        n_needed: number of samples to collect
        min_unique_classes: minimum unique labels required per mask (e.g. 2 for binary)
        max_passes: number of full dataloader passes allowed before giving up

    Returns:
        images_sel: [N, C, H, W] tensor on `device`
        masks_sel:  [N, H, W] tensor on `device`
        Prints a warning if fewer than `n_needed` qualifying samples are found.
    """
    images_buf, masks_buf = [], []
    passes = 0

    # Iterate multiple times through the dataloader if needed
    while passes < max_passes and len(images_buf) < n_needed:
        for images, masks in dataloader:
            # Move tensors to GPU/CPU as specified
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Ensure masks have shape [B, H, W] (squeeze channel dim if present)
            if masks.ndim == 4 and masks.size(1) == 1:
                masks = masks.squeeze(1)
            elif masks.ndim != 3:
                raise ValueError(f"Expected masks shape [B,H,W] or [B,1,H,W], got {tuple(masks.shape)}")

            # Loop through samples in this batch
            for i in range(masks.size(0)):
                if len(images_buf) >= n_needed:
                    break
                # Check if mask has at least `min_unique_classes` unique labels
                if torch.unique(masks[i]).numel() >= min_unique_classes:
                    images_buf.append(images[i])
                    masks_buf.append(masks[i])

            # Stop early if enough samples have been collected
            if len(images_buf) >= n_needed:
                break
        passes += 1

    # Warn if not enough qualifying samples were found
    if len(images_buf) < n_needed:
        print(f"⚠️ Only found {len(images_buf)} samples with ≥{min_unique_classes} classes "
              f"after {passes} pass(es). Returning what was found.")

    # Stack all collected samples into tensors (or None if empty)
    images_sel = torch.stack(images_buf, dim=0) if images_buf else None
    masks_sel = torch.stack(masks_buf, dim=0) if masks_buf else None
    return images_sel, masks_sel


# ---------------------------------------------------------------------
# Visualization function: visualize_segmentation
# ---------------------------------------------------------------------
def visualize_segmentation(
    model,
    dataloader,
    epoch,
    device=None,
    n=6,
    overlay=False,
    alpha=0.5,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    class_colors=((0, 0, 0), (1, 0, 0)),  # black for background, red for class-1
):
    """
    Visualize predictions for a binary (2-class) segmentation model.
    Displays or saves side-by-side comparisons of:
    - Original image
    - Ground-truth mask
    - Model prediction (optionally overlaid)

    Args:
        model: trained PyTorch segmentation model
        dataloader: yields (images, masks)
        epoch: current training epoch (used for filename)
        device: torch.device or string ('cuda' or 'cpu'); auto-detects if None
        n: number of samples to visualize
        overlay: if True, overlay masks on top of the image
        alpha: transparency for overlay visualization
        mean, std: normalization parameters used for denormalization (ImageNet defaults)
        class_colors: RGB color tuples for each class in [0,1] range
    """
    # Store whether model was in training mode, to restore later
    model_was_training = model.training
    model.eval()

    # Select device automatically if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Collect samples containing at least two unique classes
    images, masks = collect_two_class_samples(dataloader, device, n_needed=6, max_passes=5)

    # Disable gradient computation for inference
    with torch.no_grad():
        logits = model(images)

        # Handle multi-class case with 2 output channels
        if logits.dim() == 4 and logits.size(1) == 2:
            preds = torch.argmax(logits, dim=1)  # [B, H, W]
        else:
            # Handle binary case with single-channel logits
            if logits.dim() == 4 and logits.size(1) == 1:
                probs = torch.sigmoid(logits[:, 0, ...])
            elif logits.dim() == 3:
                probs = torch.sigmoid(logits)
            else:
                raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")
            preds = (probs > 0.5).long()  # threshold at 0.5 for binary prediction

    # -----------------------------------------------------------------
    # Denormalize input images (undo mean/std normalization)
    # -----------------------------------------------------------------
    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 3, 1, 1)
    imgs_disp = torch.clamp(images * std + mean, 0, 1)  # bring back to [0,1] range

    # Convert tensors to NumPy for plotting
    imgs_np = imgs_disp[:n].cpu().permute(0, 2, 3, 1).numpy()
    gts_np = masks[:n].cpu().numpy().astype(np.int64)
    preds_np = preds[:n].cpu().numpy().astype(np.int64)

    # Define color map for masks (e.g. background=black, target=red)
    cm = ListedColormap(np.array(class_colors))

    # Define layout and titles
    if overlay:
        cols = 3
        titles = ["Image", "GT overlay", "Pred overlay"]
    else:
        cols = 3
        titles = ["Image", "Ground Truth", "Prediction"]

    rows = min(n, imgs_np.shape[0])
    plt.figure(figsize=(cols * 4, rows * 3))

    # -----------------------------------------------------------------
    # Plot each sample in a grid
    # -----------------------------------------------------------------
    for i in range(rows):
        # (1) Original RGB image
        ax = plt.subplot(rows, cols, i * cols + 1)
        ax.imshow(imgs_np[i])
        ax.set_title(titles[0])
        ax.axis("off")

        # (2) Ground truth mask
        ax = plt.subplot(rows, cols, i * cols + 2)
        if overlay:
            ax.imshow(imgs_np[i])
            ax.imshow(gts_np[i], cmap=cm, alpha=alpha, vmin=0, vmax=1)
        else:
            ax.imshow(gts_np[i], cmap=cm, vmin=0, vmax=1)
        ax.set_title(titles[1])
        ax.axis("off")

        # (3) Predicted mask
        ax = plt.subplot(rows, cols, i * cols + 3)
        if overlay:
            ax.imshow(imgs_np[i])
            ax.imshow(preds_np[i], cmap=cm, alpha=alpha, vmin=0, vmax=1)
        else:
            ax.imshow(preds_np[i], cmap=cm, vmin=0, vmax=1)
        ax.set_title(titles[2])
        ax.axis("off")

    plt.tight_layout()

    # Save visualization to output folder defined in training_config.py
    plt.savefig(os.path.join(OUTPUT_FOLDER_NAME, f"{epoch}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Restore model to its original (training) state if necessary
    if model_was_training:
        model.train()