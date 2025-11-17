import os
import random
import shutil
from pathlib import Path

def make_subset(cityscapes_root, output_root, train_size=500, val_size=100, seed=42):
    random.seed(seed)

    left_dir = Path(cityscapes_root) / "leftImg8bit_trainvaltest" / "leftImg8bit"
    gt_dir = Path(cityscapes_root) / "gtFine_trainvaltest" / "gtFine"

    out_root = Path(output_root)
    splits = ["train", "val"]
    sizes = {"train": train_size, "val": val_size}

    for split in splits:
        split_input = out_root / split / "input"
        split_masks = out_root / split / "masks"
        split_input.mkdir(parents=True, exist_ok=True)
        split_masks.mkdir(parents=True, exist_ok=True)

        cities = list((left_dir / split).glob("*"))
        img_mask_pairs = []

        # Select all image-mask pairs from each city
        for city in cities:
            img_files = list(city.glob("*_leftImg8bit.png"))
            for img_path in img_files:
                mask_path = gt_dir / split / city.name / img_path.name.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
                if mask_path.exists():
                    img_mask_pairs.append((img_path, mask_path))

        # If the requested number is larger than available, take all
        n_samples = min(sizes[split], len(img_mask_pairs))
        selected_pairs = random.sample(img_mask_pairs, n_samples)

        # Copy to new 'subset dataset'
        for i, (img_path, mask_path) in enumerate(selected_pairs):
            shutil.copy(img_path, split_input / f"{i:05d}.png")
            shutil.copy(mask_path, split_masks / f"{i:05d}.png")

        print(f"{split}: {len(selected_pairs)} images copied to {split_input} and {split_masks}")

# --- Example usage ---
if __name__ == "__main__":
    make_subset(
        cityscapes_root="data/cityscapes",
        output_root="data/mini-cityscape",
        train_size=500,
        val_size=100
    )
