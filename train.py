import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
from models.unet import UNet
from dataset.mini_cityscape_dataset import MiniCityscapeDataset

# -----------------------
# Metrics
# -----------------------
def compute_miou(pred, target, n_classes=32, ignore_index=255):
    pred = pred.argmax(1)  # B x H x W
    miou = 0.0
    valid_classes = 0
    for cls in range(n_classes):
        if cls == ignore_index:
            continue
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        if union > 0:
            miou += intersection / union
            valid_classes += 1
    return miou / max(valid_classes, 1)

def pixel_accuracy(pred, target, ignore_index=255):
    pred = pred.argmax(1)
    mask = target != ignore_index
    correct = (pred[mask] == target[mask]).sum().item()
    total = mask.sum().item()
    return correct / max(total, 1)

# -----------------------
# Training function
# -----------------------
def train(batch_size=1, n_classes=32, lr=1e-4, epochs=1, train_max_samples=550, test_max_samples=50, log_interval=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(
        project="cityscape_unet_training",
        config={
            "batch_size": batch_size,
            "n_classes": n_classes,
            "learning_rate": lr,
            "epochs": epochs,
            "train_max_samples": train_max_samples,
            "test_max_samples": test_max_samples
        }
    )

    # -----------------------
    # Dataset & DataLoader
    # -----------------------
    train_full = MiniCityscapeDataset(root="data/mini-cityscape", split='train', max_samples=train_max_samples)
    val_full = MiniCityscapeDataset(root="data/mini-cityscape", split='val', max_samples=test_max_samples)

    # Use Subset to limit samples
    train_dataset = torch.utils.data.Subset(train_full, list(range(len(train_full))))
    val_dataset = torch.utils.data.Subset(val_full, list(range(len(val_full))))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # -----------------------
    # Model
    # -----------------------
    model = UNet(n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    wandb.watch(model, log="all", log_freq=100)

    print("Data loaded, model ready.")

    # -----------------------
    # Training loop
    # -----------------------
    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0
        train_miou_total = 0.0
        train_pixel_acc_total = 0.0
        num_batches = 0

        for batch_idx, (imgs, masks) in enumerate(train_loader, start=1):
            imgs, masks = imgs.to(device), masks.to(device)

            predictions = model(imgs)
            loss = F.cross_entropy(predictions, masks, ignore_index=255)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            train_loss_total += loss.item()
            train_miou_total += compute_miou(predictions, masks, n_classes=n_classes)
            train_pixel_acc_total += pixel_accuracy(predictions, masks)
            num_batches += 1

            # Log batch-level metrics
            if batch_idx % log_interval == 0:
                # Compute epoch averages
                train_loss = train_loss_total / num_batches
                train_miou = train_miou_total / num_batches
                train_pixel_acc = train_pixel_acc_total / num_batches
                print(f"Batch {batch_idx}/{len(train_loader)}, "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Train mIoU: {train_miou:.4f}, "
                      f"Train Pixel Acc: {train_pixel_acc:.4f}")
                
                wandb.log({
                    "epoch": epoch + 1,
                    "batch": batch_idx,
                    "train_loss": train_loss_total / num_batches,
                    "train_mIoU": train_miou_total / num_batches,
                    "train_pixel_acc": train_pixel_acc
                })


        # -----------------------
        # Validation
        # -----------------------
        model.eval()
        print("Evaluating...")
        val_loss_total = 0.0
        val_miou_total = 0.0
        val_pixel_acc_total = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                predictions = model(imgs)
                loss = F.cross_entropy(predictions, masks, ignore_index=255)

                val_loss_total += loss.item()
                val_miou_total += compute_miou(predictions, masks, n_classes=n_classes)
                val_pixel_acc_total += pixel_accuracy(predictions, masks)
                num_val_batches += 1

        val_loss = val_loss_total / num_val_batches
        val_miou = val_miou_total / num_val_batches
        val_pixel_acc = val_pixel_acc_total / num_val_batches

        print(f"Validation Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}, Pixel Acc: {val_pixel_acc:.4f}")

        # Log epoch-level metrics
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mIoU": train_miou,
            "val_mIoU": val_miou,
            "pixel_acc": val_pixel_acc
        })

    # -----------------------
    # Save model
    # -----------------------
    torch.save(model.state_dict(), "unet_weights.pth")
    print("Model weights saved to unet_weights.pth")


if __name__ == "__main__":
    train()
