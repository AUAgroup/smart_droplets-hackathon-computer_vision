import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.functional import jaccard_index
from hackathon_utils.data_config import *               # Imports dataset-specific constants (e.g., NUM_CLASSES)
from hackathon_utils.auxiliar import mean_values_by_key  # Helper for averaging metrics across classes/dicts

# Select computation device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# Function: multiclass_jaccard
# ---------------------------------------------------------------------
def multiclass_jaccard(
    pred: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 3,
    *,
    from_logits: bool = True,
    ignore_index: int | None = None,
    round_ndigits: int = 4,
):
    """
    Computes mean Intersection over Union (IoU, or Jaccard Index)
    and per-class IoU for a multi-class segmentation task.

    Args:
        pred: 
            - If from_logits=True → [B, C, H, W] (logits or class probabilities).
            - If from_logits=False → [B, H, W] (integer class indices).
        target: [B, H, W] tensor of class indices.
        num_classes: Total number of classes in segmentation output.
        from_logits: Whether to convert raw logits/probabilities to discrete labels.
        ignore_index: Optional class index to exclude from IoU calculation.
        round_ndigits: Number of decimal digits to round output values.

    Returns:
        mean_iou (float): Mean IoU across all valid classes.
        iou_per_class (dict[int, float]): IoU for each class index.
    """

    # ---------------------------------------------------------------
    # 1. Validate and convert predictions
    # ---------------------------------------------------------------
    if from_logits:
        # Expecting model output as [B, C, H, W]
        if pred.ndim != 4:
            raise ValueError(f"`pred` must be [B,C,H,W] when from_logits=True, got {tuple(pred.shape)}")
        # Convert to discrete class labels via argmax across channels
        pred_labels = pred.argmax(dim=1)  # → [B, H, W]
    else:
        # If already label maps, verify shape
        pred_labels = pred
        if pred_labels.ndim != 3:
            raise ValueError(f"`pred` must be [B,H,W] when from_logits=False, got {tuple(pred.shape)}")

    if target.ndim != 3:
        raise ValueError(f"`target` must be [B,H,W], got {tuple(target.shape)}")

    # ---------------------------------------------------------------
    # 2. Compute per-class IoU using torchmetrics (stateless)
    # ---------------------------------------------------------------
    # average='none' → returns one IoU value per class
    iou_per_class_tensor = jaccard_index(
        preds=pred_labels,
        target=target,
        task="multiclass",
        num_classes=num_classes,
        average="none",
        ignore_index=ignore_index,
    )

    # Replace NaN values (when class absent in batch) with 0.0 to avoid errors
    iou_per_class_tensor = torch.nan_to_num(iou_per_class_tensor, nan=0.0)

    # ---------------------------------------------------------------
    # 3. Handle optional ignore_index (exclude specified class)
    # ---------------------------------------------------------------
    classes = list(range(num_classes))
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        # Create boolean mask to remove ignored class from mean computation
        valid_mask = torch.ones(num_classes, dtype=torch.bool, device=iou_per_class_tensor.device)
        valid_mask[ignore_index] = False
        valid_classes = [c for c in classes if c != ignore_index]
        iou_valid = iou_per_class_tensor[valid_mask]
    else:
        valid_classes = classes
        iou_valid = iou_per_class_tensor

    # ---------------------------------------------------------------
    # 4. Aggregate metrics: mean and per-class
    # ---------------------------------------------------------------
    mean_iou = float(iou_valid.mean().item())
    iou_per_class = {
        cls: round(float(iou_per_class_tensor[cls].item()), round_ndigits)
        for cls in classes
    }

    return round(mean_iou, round_ndigits), iou_per_class


# ---------------------------------------------------------------------
# Function: evaluate
# ---------------------------------------------------------------------
def evaluate(model, loader):
    """
    Evaluates a segmentation model on a dataset and computes:
    - Average loss
    - Mean IoU (Jaccard)
    - IoU per class (averaged across all batches)

    Args:
        model: Trained PyTorch segmentation model.
        loader: DataLoader providing (images, masks) batches.

    Returns:
        epoch_loss (float): Mean cross-entropy loss over the dataset.
        avg_jaccard_scores (float): Mean Jaccard Index (mIoU).
        avg_jaccard_per_class (dict[int, float]): Mean IoU per class.
    """
    model.eval()  # Disable dropout/batchnorm updates

    running_loss = 0.0
    jaccard_scores = []           # Per-batch mean IoU
    all_jaccards_per_class = []   # Per-batch per-class IoU dicts

    # ---------------------------------------------------------------
    # 1. Batch-wise evaluation loop
    # ---------------------------------------------------------------
    with torch.no_grad():  # No gradient computation for evaluation
        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass through model
            outputs = model(images)

            # Compute segmentation loss (multiclass cross-entropy)
            loss = F.cross_entropy(outputs, masks)
            # Alternative (for binary segmentation):
            # loss = F.binary_cross_entropy_with_logits(outputs, masks.unsqueeze(1).float())

            # Convert logits → predicted labels
            sample_preds = outputs.argmax(dim=1)

            # Compute mean IoU and per-class IoU for this batch
            mjaccard, ijaccard_per_class = multiclass_jaccard(
                outputs, masks, num_classes=NUM_CLASSES
            )

            # Accumulate per-batch results
            all_jaccards_per_class.append(ijaccard_per_class)
            jaccard_scores.append(mjaccard)

            # Weighted accumulation of total loss
            running_loss += loss.item() * images.size(0)

        # -----------------------------------------------------------
        # 2. Aggregate results across all batches
        # -----------------------------------------------------------
        epoch_loss = running_loss / len(loader.dataset)  # Mean dataset loss
        avg_jaccard_scores = np.mean(jaccard_scores)     # Mean of batch mIoU values
        avg_jaccard_per_class = mean_values_by_key(all_jaccards_per_class)  # Average dict of per-class IoUs

    # Return metrics for this epoch (used in train_model loop)
    return epoch_loss, avg_jaccard_scores, avg_jaccard_per_class