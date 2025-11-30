import os 
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class MiniCityscapeDataset(Dataset):
    def __init__(self, root, split='train', transform=None, max_samples=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.images = sorted(os.listdir(os.path.join(root, split, 'input')))
        self.masks = sorted(os.listdir(os.path.join(root, split, 'masks')))

        if max_samples is not None:
            self.images = self.images[:max_samples]
            self.masks = self.masks[:max_samples]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx, target_size=(512, 256)):
        img_path = os.path.join(self.root, self.split, 'input', self.images[idx])
        mask_path = os.path.join(self.root, self.split, 'masks', self.masks[idx])

        image = np.array(Image.open(img_path).convert("RGB").resize(target_size))
        mask  = np.array(Image.open(mask_path).resize(target_size, resample=Image.NEAREST))

        # convert to tensors
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
        mask = torch.tensor(mask).long()

        # Ignore all labels >= 32
        mask[mask >= 32] = 255
        return image, mask