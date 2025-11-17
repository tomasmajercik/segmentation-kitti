import os 
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class MiniCityscapeDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.images = sorted(os.listdir(os.path.join(root, split, 'input')))
        self.masks = sorted(os.listdir(os.path.join(root, split, 'masks')))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.split, 'input', self.images[idx])
        mask_path = os.path.join(self.root, self.split, 'masks', self.masks[idx])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        # convert to tensors
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
        mask = torch.tensor(mask).long()

        return image, mask