import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.functional import jaccard_index
from hackathon_utils.data_config import *
from hackathon_utils.auxiliar import mean_values_by_key

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    Compute mean IoU and IoU per class for multiclass segmentation.

    Args:
        pred:   If from_logits=True -> [B, C, H, W] (logits or probabilities).
                If from_logits=False -> [B, H, W] (class indices).
        target: [B, H, W] class indices.
        num_classes: Number of classes.
        from_logits: Whether `pred` is logits/probabilities (True) or label maps (False).
        ignore_index: Class to ignore in IoU computation.
        round_ndigits: Rounding for returned floats.

    Returns:
        mean_iou (float): mean IoU over classes (excluding ignore_index if set).
        iou_per_class (dict[int, float]): per-class IoU.
    """
    
    # Convert predictions to class indices if needed
    if from_logits:
        if pred.ndim != 4:
            raise ValueError(f"`pred` must be [B,C,H,W] when from_logits=True, got {tuple(pred.shape)}")
        pred_labels = pred.argmax(dim=1)  # [B, H, W]
    else:
        pred_labels = pred
        if pred_labels.ndim != 3:
            raise ValueError(f"`pred` must be [B,H,W] when from_logits=False, got {tuple(pred.shape)}")

    if target.ndim != 3:
        raise ValueError(f"`target` must be [B,H,W], got {tuple(target.shape)}")

    # torchmetrics functional API (no state carried across calls)
    # average='none' -> per-class IoU
    iou_per_class_tensor = jaccard_index(
        preds=pred_labels,
        target=target,
        task="multiclass",
        num_classes=num_classes,
        average="none",
        ignore_index=ignore_index,
    )

    # Replace NaNs (classes not present) with 0.0 for stability
    iou_per_class_tensor = torch.nan_to_num(iou_per_class_tensor, nan=0.0)

    # Build per-class dict (skip ignore_index if set)
    classes = list(range(num_classes))
    if ignore_index is not None and 0 <= ignore_index < num_classes:
        valid_mask = torch.ones(num_classes, dtype=torch.bool, device=iou_per_class_tensor.device)
        valid_mask[ignore_index] = False
        valid_classes = [c for c in classes if c != ignore_index]
        iou_valid = iou_per_class_tensor[valid_mask]
    else:
        valid_classes = classes
        iou_valid = iou_per_class_tensor

    mean_iou = float(iou_valid.mean().item())
    iou_per_class = {cls: round(float(iou_per_class_tensor[cls].item()), round_ndigits) for cls in classes}

    return round(mean_iou, round_ndigits), iou_per_class

def evaluate(model, loader):
    model.eval()

    running_loss = 0.0
    dice_losses = []
    iou_scores = []
    all_ious_per_class = []
    all_jaccards_per_class = []
    jaccard_scores = []
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = F.cross_entropy(outputs, masks)
            #loss = F.binary_cross_entropy_with_logits(outputs, masks.unsqueeze(1).float())
            sample_preds = outputs.argmax(dim=1)

            mjaccard, ijaccard_per_class = multiclass_jaccard(outputs, masks, num_classes=NUM_CLASSES)
            all_jaccards_per_class.append(ijaccard_per_class)
            jaccard_scores.append(
                mjaccard
            )
    
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(loader.dataset)
        #logger.debug(f"IOU Scores: {iou_scores}")
        avg_jaccard_scores = np.mean(jaccard_scores)
        #logger.debug(f"Jaccard Scores: {jaccard_scores}")
        #logger.debug(f"IOU Per Class: {avg_iou_per_class}")
        avg_jaccard_per_class = mean_values_by_key(all_jaccards_per_class)
        #logger.debug(f"Jaccard Per Class: {avg_jaccard_per_class}")

                
    return epoch_loss, avg_jaccard_scores, avg_jaccard_per_class