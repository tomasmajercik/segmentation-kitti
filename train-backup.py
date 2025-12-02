import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
# from models.unet import UNet
from models.unet_mini import UNet
from dataset.mini_cityscape_dataset import MiniCityscapeDataset


def set_seed(seed=42):
    np.random.seed(seed)                    # NumPy random
    torch.manual_seed(seed)                 # PyTorch CPU
    torch.cuda.manual_seed(seed)            # PyTorch GPU
    torch.cuda.manual_seed_all(seed)        # Ak je viac GPU
    torch.backends.cudnn.deterministic = True  # Deterministic mode (pomalejšie)
    torch.backends.cudnn.benchmark = False    # Zabraňuje nelineárnym variáciám

set_seed(42)
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

def fast_hist(pred, label, n_class):
    mask = (label >= 0) & (label < n_class)
    hist = torch.bincount(
        n_class * label[mask].to(torch.int64) + pred[mask].to(torch.int64),
        minlength=n_class**2
    ).reshape(n_class, n_class)
    return hist


# -----------------------
# Training function
# -----------------------
def train(
        batch_size=8, 
        n_classes=32, 
        lr=3e-4,
        weight_decay=1e-4,
        epochs=5, 
        train_max_samples=550, 
        test_max_samples=50, 
        log_interval=10,
        label_smoothing=0.05,
        eta_min=1e-6
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb.init(
        project="cityscape_unet_training",
        group="fix-jumping",
        name=f"small-adamW",
        config={
            "batch_size": batch_size,
            "n_classes": n_classes,
            "learning_rate": lr,
            "epochs": epochs,
            "train_max_samples": train_max_samples,
            "test_max_samples": test_max_samples,
            "model": "smallUnet",
            "optimizer": "adamW",
            "eta_min": eta_min,
            "weight_decay": weight_decay
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
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=eta_min
    )


    wandb.watch(model, log="all", log_freq=100)

    print("Data loaded, model ready.")

    # -----------------------
    # Training loop
    # -----------------------
    for epoch in range(epochs):
        model.train()
        train_loader_iter = iter(train_loader)
        train_loss_total = 0.0
        train_miou_total = 0.0
        train_pixel_acc_total = 0.0
        num_batches = 0
        hist_sum = torch.zeros((n_classes, n_classes), dtype=torch.int64).to(device)


        for batch_idx, (imgs, masks) in enumerate(train_loader_iter, start=1):
            imgs, masks = imgs.to(device), masks.to(device)

            predictions = model(imgs)
            loss = F.cross_entropy(predictions, masks, ignore_index=255, label_smoothing=label_smoothing)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            train_loss_total += loss.item()
            # train_miou_total += compute_miou(predictions, masks, n_classes=n_classes)
            train_pixel_acc_total += pixel_accuracy(predictions, masks)
            num_batches += 1

            pred_labels = predictions.argmax(1)
            hist_sum += fast_hist(pred_labels, masks, n_classes)

            # Log batch-level metrics
            if batch_idx % log_interval == 0:
                # Compute epoch averages
                print(f"Sample {batch_idx}/{len(train_loader)}, "
                      f"Epoch: {epoch+1}/{epochs}, "
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

            hist_np = hist_sum.cpu().numpy()
            iou = np.diag(hist_np) / (hist_np.sum(1) + hist_np.sum(0) - np.diag(hist_np) + 1e-7)
            train_miou = np.nanmean(iou)
            train_loss = train_loss_total / num_batches
            train_pixel_acc = train_pixel_acc_total / num_batches

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
                loss = F.cross_entropy(predictions, masks, ignore_index=255, label_smoothing=label_smoothing)

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
        scheduler.step()


    # -----------------------
    # Save model
    # -----------------------
    torch.save(model.state_dict(), "unet_weights.pth")
    print("Model weights saved to unet_weights.pth")


if __name__ == "__main__":
    train()
