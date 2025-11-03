import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from hackathon_utils.data_config import *
from hackathon_utils.training_config import BATCH_SIZE


class VOCSegmentationDataset(Dataset):
    """
    Unified dataset class for binary segmentation.
    Handles:
      - pairing image/mask files
      - resizing
      - normalization
      - binarization of masks (0/255 → {0,1})
    """

    def __init__(
        self,
        img_dir,
        mask_dir,
        preprocess_input=None,
        img_size=512,
    ):
        self.img_paths = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
        ])
        self.mask_paths = sorted([
            os.path.join(mask_dir, f)
            for f in os.listdir(mask_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))
        ])

        if len(self.img_paths) != len(self.mask_paths):
            raise ValueError(f"Image/mask count mismatch: {len(self.img_paths)} vs {len(self.mask_paths)}")

        # Load normalization parameters
        if preprocess_input is None:
            preprocess_input = {
                'mean': (0.485, 0.456, 0.406),
                'std': (0.229, 0.224, 0.225)
            }
        self.mean = preprocess_input['mean']
        self.std = preprocess_input['std']
        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # --- Load images ---
        image = Image.open(self.img_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx]).convert('L')  # grayscale 0/255

        # --- Resize ---
        image = TF.resize(image, (self.img_size, self.img_size), interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (self.img_size, self.img_size), interpolation=InterpolationMode.NEAREST)

        # --- Transform to tensors ---
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)

        # --- Binarize and convert mask ---
        mask_np = np.array(mask, dtype=np.uint8)
        #mask_np = (mask_np > 127).astype(np.uint8)
        mask = torch.from_numpy(mask_np).long()

        return image, mask


def get_dataloaders(preprocess_input=None):
    """
    Create train/val dataloaders using unified dataset class.
    """
    train_dataset = VOCSegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, preprocess_input)
    val_dataset = VOCSegmentationDataset(VAL_IMG_DIR, VAL_MASK_DIR, preprocess_input)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader
