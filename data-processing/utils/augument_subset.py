import random
from pathlib import Path
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

# import kornia.augmentation as K # alternative augmentations (optional) for rain, fog, etc(not really good quality).

"""
This script augments a subset of images in a dataset by applying random transformations such as rotation, scaling, flipping, and color jittering.
It saves the augmented images. It can also delete previously augmented images.
It is designed to broaden the diversity of training data for machine learning tasks and thus improve model robustness.
"""

def augument_subset(dataset_root, percent=0.3, splits=["train"], seed=42):
    random.seed(seed)

    # Color jitter for RGB image only
    color_jitter = transforms.ColorJitter(
        brightness=random.randint(0, 4) * 0.2,
        contrast=random.randint(0, 4) * 0.2,
        saturation=random.randint(0, 4) * 0.2
    )

    # Optional: Kornia augmentations (disabled by default)
    # rain_aug = K.RandomRain(
    #     drop_height=(10, 20),
    #     drop_width=(1, 2),
    #     number_of_drops=(500, 1000),
    #     p=0.5
    # )

    def apply_transforms(img, mask):
        # Horizontal flip
        if random.random() < 0.5:
            img = F.hflip(img)
            mask = F.hflip(mask)

        # Rotation
        angle = random.uniform(-10, 10)
        img = F.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        mask = F.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)

        # Scaling
        scale = random.uniform(0.8, 1.2)
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        img = F.resize(img, (new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR)
        mask = F.resize(mask, (new_h, new_w), interpolation=transforms.InterpolationMode.NEAREST)

        # Random crop back to original size
        if new_w >= w and new_h >= h:
            left = random.randint(0, new_w - w)
            top = random.randint(0, new_h - h)
            img = F.crop(img, top, left, h, w)
            mask = F.crop(img, top, left, h, w)
        else:
            # If scaled down, pad back to original size
            img = F.resize(img, (h, w), interpolation=transforms.InterpolationMode.BILINEAR)
            mask = F.resize(mask, (h, w), interpolation=transforms.InterpolationMode.NEAREST)

        return img, mask

    for split in splits:
        split_input = Path(dataset_root) / split / "input"
        split_masks = Path(dataset_root) / split / "masks"

        img_files = sorted([f for f in split_input.glob("*.png") if "_aug" not in f.name])
        n_original = len(img_files)
        n_to_generate = int(n_original * percent)

        print(f"{split}: original={n_original}, generating={n_to_generate}")

        for i in range(n_to_generate):
            img_path = random.choice(img_files)
            mask_path = split_masks / img_path.name

            img = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path)

            # Apply geometric transforms
            img_aug, mask_aug = apply_transforms(img, mask)

            # Apply color jitter (PIL)
            img_aug = color_jitter(img_aug)

            # Optional: Apply rain if enabled
            # img_tensor = transforms.ToTensor()(img_aug).unsqueeze(0)
            # img_tensor = rain_aug(img_tensor)
            # img_aug = transforms.ToPILImage()(img_tensor.squeeze(0))

            # Save
            aug_name = f"{img_path.stem}_aug_{i:05d}.png"
            img_aug.save(split_input / aug_name)
            mask_aug.save(split_masks / aug_name)

        print(f"{split}: done ({n_to_generate} new samples)")


def delete_augmented(dataset_root, splits=["train"]):
    # Delete previously augmented images
    for split in splits:
        split_input = Path(dataset_root) / split / "input"
        split_masks = Path(dataset_root) / split / "masks"

        for img_file in split_input.glob("*_aug_*.png"):
            img_file.unlink()

        for mask_file in split_masks.glob("*_aug_*.png"):
            mask_file.unlink()

        print(f"{split}: deleted all augmented images and masks")


if __name__ == "__main__":
    delete_augmented("data/mini-cityscape")
    augument_subset(
        dataset_root="data/mini-cityscape",
        percent=0.1
    )