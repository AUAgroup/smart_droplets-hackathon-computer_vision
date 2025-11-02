import os
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm


def segment_images(model, input_folder, output_folder, device=None, threshold=0.5):
    """
    Applies a semantic segmentation model to all images in a folder and saves predicted masks.

    Args:
        model: PyTorch segmentation model.
        input_folder (str): Folder containing input images.
        output_folder (str): Folder to save predicted masks.
        device: torch.device (default: cuda if available).
        threshold (float): Threshold for binary segmentation.
    """
    # ---------------------------
    # Setup
    # ---------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_folder, exist_ok=True)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    image_paths = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp"))
    ]

    # ---------------------------
    # Inference loop
    # ---------------------------
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Segmenting images"):
            # Load and preprocess
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]

            # Forward pass
            output = model(img_tensor)

            # Handle binary or multiclass outputs
            if output.ndim == 4 and output.shape[1] > 1:
                pred_mask = torch.argmax(output, dim=1)[0].cpu().numpy().astype("uint8")
            else:
                prob = torch.sigmoid(output)[0, 0]
                pred_mask = (prob > threshold).cpu().numpy().astype("uint8")

            # Save mask
            mask_img = Image.fromarray(pred_mask * 255)  # convert to visible grayscale
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_img.save(os.path.join(output_folder, f"{base_name}.png"))

    print(f"✅ Saved predicted masks in: {output_folder}")

import os
import torch
import numpy as np
from PIL import Image
from torchmetrics.functional import jaccard_index


def evaluate_iou(true_dir, pred_dir, num_classes=2, device=None):
    """
    Computes mean IoU (mIoU) and IoU per class between true and predicted masks.

    Args:
        true_dir (str): Folder with ground-truth masks.
        pred_dir (str): Folder with predicted masks.
        num_classes (int): Number of classes (e.g., 2 for binary).
        device (torch.device, optional): Computation device.

    Returns:
        mean_iou (float): Mean IoU across all classes.
        iou_per_class (dict): IoU per class {class_idx: value}.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Collect and sort mask file paths
    true_files = sorted([
        f for f in os.listdir(true_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
    ])
    pred_files = sorted([
        f for f in os.listdir(pred_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
    ])
    assert len(true_files) == len(pred_files), "Mismatch between GT and predicted masks!"

    preds, targets = [], []
    for gt_name, pr_name in zip(true_files, pred_files):
        gt_path = os.path.join(true_dir, gt_name)
        pr_path = os.path.join(pred_dir, pr_name)

        gt = np.array(Image.open(gt_path))
        pr = np.array(Image.open(pr_path))

        # Normalize (handle 0/255 masks)
        if gt.max() > 1:
            gt = (gt > 127).astype(np.uint8)
        if pr.max() > 1:
            pr = (pr > 127).astype(np.uint8)

        preds.append(torch.tensor(pr, dtype=torch.int64))
        targets.append(torch.tensor(gt, dtype=torch.int64))

    preds = torch.stack(preds).to(device)
    targets = torch.stack(targets).to(device)

    # Compute per-class IoU
    iou_per_class_tensor = jaccard_index(
        preds=preds,
        target=targets,
        task="multiclass",     # even for binary (0 and 1)
        num_classes=num_classes,
        average="none"         # keep per-class scores
    )

    # Handle NaN (class not present)
    iou_per_class_tensor = torch.nan_to_num(iou_per_class_tensor, nan=0.0)

    # Mean IoU and dictionary
    mean_iou = float(iou_per_class_tensor.mean().item())
    iou_per_class = {
        cls: round(float(iou_per_class_tensor[cls].item()), 4)
        for cls in range(num_classes)
    }

    print("✅ Evaluation complete")
    print(f"Mean IoU: {mean_iou:.4f}")
    for cls, val in iou_per_class.items():
        print(f"Class {cls}: IoU = {val:.4f}")

    return mean_iou, iou_per_class