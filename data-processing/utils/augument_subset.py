import random
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

def augument_subset(dataset_root, percent=0.3, splits=["train", "val"], seed=42):
    random.seed(seed)

    color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)

    def apply_transforms(img, mask):
        if random.random() < 0.5:
            img = F.hflip(img)
            mask = F.hflip(mask)

        angle = random.uniform(-10, 10)
        img = F.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR)
        mask = F.rotate(mask, angle, interpolation=transforms.InterpolationMode.NEAREST)

        scale = random.uniform(0.8, 1.2)
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)
        img = F.resize(img, (new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR)
        mask = F.resize(mask, (new_h, new_w), interpolation=transforms.InterpolationMode.NEAREST)

        left = random.randint(0, max(0, new_w - w))
        top = random.randint(0, max(0, new_h - h))
        img = F.crop(img, top, left, h, w)
        mask = F.crop(mask, top, left, h, w)

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

            img_aug, mask_aug = apply_transforms(img, mask)
            img_aug = color_jitter(img_aug)

            aug_name = f"{img_path.stem}_aug_{i:05d}.png"
            img_aug.save(split_input / aug_name)
            mask_aug.save(split_masks / aug_name)

        print(f"{split}: done ({n_to_generate} new samples)")

from pathlib import Path

def delete_augmented(dataset_root, splits=["train", "val"]):
    for split in splits:
        split_input = Path(dataset_root) / split / "input"
        split_masks = Path(dataset_root) / split / "masks"

        # delete images
        for img_file in split_input.glob("*_aug_*.png"):
            img_file.unlink()
        # delete masks
        for mask_file in split_masks.glob("*_aug_*.png"):
            mask_file.unlink()

        print(f"{split}: deleted all augmented images and masks")
    

if __name__ == "__main__":
    delete_augmented("data/cityscapes")
    augument_subset(
        dataset_root="data/cityscapes",
        percent=0.1
    )