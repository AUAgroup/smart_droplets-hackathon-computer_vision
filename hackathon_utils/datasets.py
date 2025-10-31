import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from hackathon_utils.data_config import *
from hackathon_utils.training_config import BATCH_SIZE

class VOCBinarySegmentation(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_paths  = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # load
        image = Image.open(self.img_paths[idx]).convert('RGB')
        mask  = Image.open(self.mask_paths[idx]).convert('L')  # 0/1 mask

        return image, mask

# 3. Define a custom Dataset wrapper for applying transforms to subsets
class SubsetWithTransforms(Dataset):
    def __init__(self, subset, 
                 preprocess_input = None):
        self.subset = subset
        self.img_size = 512 #preprocess_input['input_size'][-1]
        self.preprocess_input = preprocess_input


    def __getitem__(self, index):
        image, mask = self.subset[index]
        # Random Crop (need to get crop parameters)
        #if not self.use_colour_augmentations and not self.use_geometric_augmentations:
        image = TF.resize(image, (self.img_size, self.img_size), interpolation=Image.BILINEAR)
        mask  = TF.resize(mask, (self.img_size, self.img_size), interpolation=Image.NEAREST)
        
        image = TF.to_tensor(image)
        #if self.use_normalization:
        image = TF.normalize(image, self.preprocess_input['mean'], 
                             self.preprocess_input['std'])

        mask_array = np.array(mask, dtype=np.uint8)   
        mask  = torch.from_numpy(mask_array).long()
        
        return image, mask

    def __len__(self):
        return len(self.subset)

def get_dataloaders(
    preprocess_input=None,
):
    train_dataset = VOCBinarySegmentation(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    val_dataset = VOCBinarySegmentation(VAL_IMG_DIR, VAL_MASK_DIR)

    train_dataset = SubsetWithTransforms(
        train_dataset,
        preprocess_input=preprocess_input,
    )
    val_dataset = SubsetWithTransforms(
        val_dataset,
        preprocess_input=preprocess_input,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    # report_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    return train_loader, val_loader