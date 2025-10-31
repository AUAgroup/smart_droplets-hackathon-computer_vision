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