import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

def collect_two_class_samples(
    dataloader,
    device,
    n_needed: int = 6,
    min_unique_classes: int = 2,
    max_passes: int = 3,  # how many full passes over the dataloader before giving up
):
    """
    Collect N samples (images, masks) where each mask contains at least `min_unique_classes` classes.

    Args:
        dataloader: yields (images [B, C, H, W], masks [B, H, W] or [B,1,H,W])
        device: torch.device or str
        n_needed: number of qualifying samples to return
        min_unique_classes: minimum unique labels required in each mask (>=2 for binary)
        max_passes: maximum full passes over dataloader to try before stopping

    Returns:
        images_sel: [N, C, H, W] tensor on `device`
        masks_sel:  [N, H, W] tensor on `device`
        note: if fewer than N found, returns whatever was found (N' < N) and prints a warning
    """
    images_buf, masks_buf = [], []
    passes = 0

    while passes < max_passes and len(images_buf) < n_needed:
        for images, masks in dataloader:
            # move to device
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # ensure masks are [B, H, W]
            if masks.ndim == 4 and masks.size(1) == 1:
                masks = masks.squeeze(1)
            elif masks.ndim != 3:
                raise ValueError(f"Expected masks shape [B,H,W] or [B,1,H,W], got {tuple(masks.shape)}")

            # filter samples in this batch
            for i in range(masks.size(0)):
                if len(images_buf) >= n_needed:
                    break
                # quick unique count per-sample
                if torch.unique(masks[i]).numel() >= min_unique_classes:
                    images_buf.append(images[i])
                    masks_buf.append(masks[i])

            if len(images_buf) >= n_needed:
                break
        passes += 1

    if len(images_buf) < n_needed:
        print(f"⚠️ Only found {len(images_buf)} samples with ≥{min_unique_classes} classes "
              f"after {passes} pass(es). Returning what was found.")

    # stack to tensors on device
    images_sel = torch.stack(images_buf, dim=0) if images_buf else None
    masks_sel = torch.stack(masks_buf, dim=0) if masks_buf else None
    return images_sel, masks_sel

def visualize_segmentation(
    model,
    dataloader,
    device=None,
    n=6,
    overlay=False,
    alpha=0.5,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    class_colors=((0, 0, 0), (1, 0, 0)),  # background (black), class-1 (red)
):
    """
    Visualize original images, ground truth masks, and predicted masks for a 2-class segmentation task.

    Args:
        model: PyTorch segmentation model.
        dataloader: Yields (images, masks). 
                    images: [B, 3, H, W] (float, normalized)
                    masks:  [B, H, W] with {0,1} or [B,1,H,W] with {0,1}
        device: torch.device or str. If None, auto-detect.
        n: number of samples to display (from the first batch).
        overlay: if True, show masks overlaid on images; otherwise show masks as standalone panels.
        alpha: overlay transparency for masks.
        mean, std: per-channel normalization used to denormalize images for display.
        class_colors: tuple of two RGB tuples in [0,1] for classes {0,1}.
    """
    model_was_training = model.training
    model.eval()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    images, masks = collect_two_class_samples(val_loader, device, n_needed=6, max_passes=5)

    with torch.no_grad():
        logits = model(images)
        # unify shape handling
        if logits.dim() == 4 and logits.size(1) == 2:
            preds = torch.argmax(logits, dim=1)  # [B,H,W]
        else:
            # assume binary logits in shape [B,1,H,W] or [B,H,W]
            if logits.dim() == 4 and logits.size(1) == 1:
                probs = torch.sigmoid(logits[:, 0, ...])
            elif logits.dim() == 3:
                probs = torch.sigmoid(logits)
            else:
                raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")
            preds = (probs > 0.5).long()

    # denormalize images to [0,1] for display
    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 3, 1, 1)
    imgs_disp = torch.clamp(images * std + mean, 0, 1)

    # to cpu numpy
    imgs_np = imgs_disp[:n].cpu().permute(0, 2, 3, 1).numpy()
    gts_np = masks[:n].cpu().numpy().astype(np.int64)
    preds_np = preds[:n].cpu().numpy().astype(np.int64)

    # color maps for masks
    cm = ListedColormap(np.array(class_colors))

    # figure layout: rows = n, cols = 3 (Image | GT | Pred) or overlay variant
    if overlay:
        cols = 3
        titles = ["Image", "GT overlay", "Pred overlay"]
    else:
        cols = 3
        titles = ["Image", "Ground Truth", "Prediction"]

    rows = min(n, imgs_np.shape[0])
    plt.figure(figsize=(cols * 4, rows * 3))

    for i in range(rows):
        # original image
        ax = plt.subplot(rows, cols, i * cols + 1)
        ax.imshow(imgs_np[i])
        ax.set_title(titles[0])
        ax.axis("off")

        # ground truth
        ax = plt.subplot(rows, cols, i * cols + 2)
        if overlay:
            ax.imshow(imgs_np[i])
            ax.imshow(gts_np[i], cmap=cm, alpha=alpha, vmin=0, vmax=1)
        else:
            ax.imshow(gts_np[i], cmap=cm, vmin=0, vmax=1)
        ax.set_title(titles[1])
        ax.axis("off")

        # prediction
        ax = plt.subplot(rows, cols, i * cols + 3)
        if overlay:
            ax.imshow(imgs_np[i])
            ax.imshow(preds_np[i], cmap=cm, alpha=alpha, vmin=0, vmax=1)
        else:
            ax.imshow(preds_np[i], cmap=cm, vmin=0, vmax=1)
        ax.set_title(titles[2])
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    #plt.savefig(os.path.join(output_dir, f"{model_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # restore training state
    if model_was_training:
        model.train()