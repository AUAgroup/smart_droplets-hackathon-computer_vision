import os
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from torchmetrics.functional import jaccard_index

# ---------------------------------------------------------------------
# Function: segment_images
# ---------------------------------------------------------------------
def segment_images(model, input_folder, output_folder, device=None, threshold=0.5):
    """
    Applies a trained semantic segmentation model to all images in a folder
    and saves the predicted binary or multiclass masks.

    Args:
        model (torch.nn.Module): Trained PyTorch segmentation model.
        input_folder (str): Path to folder containing input images.
        output_folder (str): Directory where predicted masks will be saved.
        device (torch.device, optional): GPU or CPU for inference (auto-detect if None).
        threshold (float): Probability threshold for binary segmentation (default: 0.5).
    """
    # ---------------------------------------------------------------
    # 1. Setup
    # ---------------------------------------------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()

    # Define preprocessing transforms (standard ImageNet normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    # Collect all valid image paths from input directory
    image_paths = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp"))
    ]

    # ---------------------------------------------------------------
    # 2. Inference loop
    # ---------------------------------------------------------------
    with torch.no_grad():  # disable gradients for faster inference
        for img_path in tqdm(image_paths, desc="Segmenting images"):
            # ---- Load and preprocess input image ----
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)  # shape: [1, 3, H, W]

            # ---- Forward pass through model ----
            output = model(img_tensor)

            # ---- Post-processing ----
            if output.ndim == 4 and output.shape[1] > 1:
                # Multiclass segmentation: take argmax across channels
                pred_mask = torch.argmax(output, dim=1)[0].cpu().numpy().astype("uint8")
            else:
                # Binary segmentation: apply sigmoid + threshold
                prob = torch.sigmoid(output)[0, 0]
                pred_mask = (prob > threshold).cpu().numpy().astype("uint8")

            # ---- Save predicted mask ----
            # Convert {0,1} mask to visible grayscale (0 or 255)
            mask_img = Image.fromarray(pred_mask * 255)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_img.save(os.path.join(output_folder, f"{base_name}.png"))

    print(f"✅ Saved predicted masks in: {output_folder}")

# ---------------------------------------------------------------------
# Function: evaluate_iou
# ---------------------------------------------------------------------
def evaluate_iou(true_dir, pred_dir, num_classes=2, device=None):
    """
    Computes mean Intersection over Union (mIoU) and per-class IoU 
    between ground-truth and predicted segmentation masks.

    Args:
        true_dir (str): Path to folder containing ground-truth masks.
        pred_dir (str): Path to folder containing predicted masks.
        num_classes (int): Number of classes (e.g., 2 for binary segmentation).
        device (torch.device, optional): Device for metric computation.

    Returns:
        mean_iou (float): Mean IoU across all classes.
        iou_per_class (dict[int, float]): IoU value per class index.
    """
    # ---------------------------------------------------------------
    # 1. Setup
    # ---------------------------------------------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Gather and sort all file names for consistent ordering
    true_files = sorted([
        f for f in os.listdir(true_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
    ])
    pred_files = sorted([
        f for f in os.listdir(pred_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
    ])
    assert len(true_files) == len(pred_files), \
        "Mismatch between number of ground-truth and predicted masks!"

    # ---------------------------------------------------------------
    # 2. Load masks and preprocess
    # ---------------------------------------------------------------
    preds, targets = [], []
    for gt_name, pr_name in zip(true_files, pred_files):
        gt_path = os.path.join(true_dir, gt_name)
        pr_path = os.path.join(pred_dir, pr_name)

        # Load ground-truth and predicted masks as NumPy arrays
        gt = np.array(Image.open(gt_path))
        pr = np.array(Image.open(pr_path))

        # Normalize: convert grayscale (0/255) to binary (0/1)
        if gt.max() > 1:
            gt = (gt > 127).astype(np.uint8)
        if pr.max() > 1:
            pr = (pr > 127).astype(np.uint8)

        # Append to list as torch tensors
        preds.append(torch.tensor(pr, dtype=torch.int64))
        targets.append(torch.tensor(gt, dtype=torch.int64))

    # Stack all masks into tensors: shape [N, H, W]
    preds = torch.stack(preds).to(device)
    targets = torch.stack(targets).to(device)

    # ---------------------------------------------------------------
    # 3. Compute IoU using torchmetrics
    # ---------------------------------------------------------------
    iou_per_class_tensor = jaccard_index(
        preds=preds,
        target=targets,
        task="multiclass",     # works for binary (0,1) or multi-class
        num_classes=num_classes,
        average="none"         # return IoU per class (no mean yet)
    )

    # Replace NaNs (caused by absent classes) with 0
    iou_per_class_tensor = torch.nan_to_num(iou_per_class_tensor, nan=0.0)

    # Compute mean IoU and build per-class dictionary
    mean_iou = float(iou_per_class_tensor.mean().item())
    iou_per_class = {
        cls: round(float(iou_per_class_tensor[cls].item()), 4)
        for cls in range(num_classes)
    }

    # ---------------------------------------------------------------
    # 4. Reporting
    # ---------------------------------------------------------------
    print("✅ Evaluation complete")
    print(f"Mean IoU: {mean_iou:.4f}")
    for cls, val in iou_per_class.items():
        print(f"Class {cls}: IoU = {val:.4f}")

    return mean_iou, iou_per_class