import torch
import numpy as np
import torch.nn.functional as F
from models.dice_loss import DiceLoss
from torch.utils.data import DataLoader

import wandb
# from models.unet import UNet
from models.unet_mini import UNet
from dataset.mini_cityscape_dataset import MiniCityscapeDataset


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

def pixel_accuracy(pred, target, ignore_index=255):
    pred = pred.argmax(1)
    mask = target != ignore_index
    correct = (pred[mask] == target[mask]).sum().item()
    total = mask.sum().item()
    return correct / max(total, 1)


def fast_hist(pred, label, n_class):
    mask = (label >= 0) & (label < n_class)
    if mask.sum() == 0:
        return torch.zeros((n_class, n_class), dtype=torch.int64, device=label.device)
    hist = torch.bincount(
        n_class * label[mask].to(torch.int64) + pred[mask].to(torch.int64),
        minlength=n_class**2
    ).reshape(n_class, n_class)
    return hist


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
        eta_min=1e-6,
        max_grad_norm=1.0
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
            "optimizer": "AdamW",
            "eta_min": eta_min,
            "weight_decay": weight_decay
        }
    )

    train_full = MiniCityscapeDataset(root="data/mini-cityscape", split='train', max_samples=train_max_samples)
    val_full = MiniCityscapeDataset(root="data/mini-cityscape", split='val', max_samples=test_max_samples)

    train_dataset = torch.utils.data.Subset(train_full, list(range(len(train_full))))
    val_dataset = torch.utils.data.Subset(val_full, list(range(len(val_full))))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = UNet(n_classes=n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    dice_loss_fn = DiceLoss(n_classes=n_classes) 

    wandb.watch(model, log="all", log_freq=100)

    print("Data loaded, model ready.")

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0
        train_pixel_acc_total = 0.0
        num_batches = 0
        hist_sum = torch.zeros((n_classes, n_classes), dtype=torch.int64).to(device)

        for batch_idx, (imgs, masks) in enumerate(train_loader, start=1):
            imgs, masks = imgs.to(device), masks.to(device)

            predictions = model(imgs)
            ce_loss = F.cross_entropy(predictions, masks, ignore_index=255, label_smoothing=label_smoothing)
            dice_loss = dice_loss_fn(predictions, masks) 
            loss = ce_loss + dice_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            train_loss_total += loss.item()
            train_pixel_acc_total += pixel_accuracy(predictions, masks)
            num_batches += 1

            pred_labels = predictions.argmax(1)
            hist_sum += fast_hist(pred_labels, masks, n_classes)

            if batch_idx % log_interval == 0:
                cur_loss = train_loss_total / num_batches
                cur_pixel = train_pixel_acc_total / num_batches
                print(f"Batch {batch_idx}/{len(train_loader)} | Epoch {epoch+1}/{epochs} | Loss {cur_loss:.4f} | PixelAcc {cur_pixel:.4f}")
                wandb.log({
                    "epoch": epoch + 1,
                    "batch": batch_idx,
                    "train_loss": cur_loss,
                    "train_pixel_acc": cur_pixel,
                    "learning_rate": optimizer.param_groups[0]['lr']
                })


        # compute train mIoU for the epoch from hist_sum
        hist_np = hist_sum.cpu().numpy().astype(np.float64)
        denom = (hist_np.sum(1) + hist_np.sum(0) - np.diag(hist_np))
        iou = np.diag(hist_np) / (denom + 1e-7)
        train_miou = np.nanmean(iou)

        train_loss = train_loss_total / max(num_batches, 1)
        train_pixel_acc = train_pixel_acc_total / max(num_batches, 1)

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_pixel_acc_total = 0.0
        num_val_batches = 0
        val_hist_sum = torch.zeros((n_classes, n_classes), dtype=torch.int64).to(device)

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                predictions = model(imgs)
                ce_loss = F.cross_entropy(predictions, masks, ignore_index=255, label_smoothing=label_smoothing)
                dice_loss = dice_loss_fn(predictions, masks)
                loss = ce_loss + dice_loss

                val_loss_total += loss.item()
                val_pixel_acc_total += pixel_accuracy(predictions, masks)
                num_val_batches += 1

                pred_labels = predictions.argmax(1)
                val_hist_sum += fast_hist(pred_labels, masks, n_classes)

        val_loss = val_loss_total / max(num_val_batches, 1)
        val_pixel_acc = val_pixel_acc_total / max(num_val_batches, 1)

        val_hist_np = val_hist_sum.cpu().numpy().astype(np.float64)
        val_denom = (val_hist_np.sum(1) + val_hist_np.sum(0) - np.diag(val_hist_np))
        val_iou = np.diag(val_hist_np) / (val_denom + 1e-7)
        val_miou = np.nanmean(val_iou)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train mIoU: {train_miou:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_mIoU": train_miou,
            "val_mIoU": val_miou,
            "pixel_acc": val_pixel_acc
        })


        scheduler.step()

    torch.save(model.state_dict(), "unet_weights.pth")
    print("Model weights saved to unet_weights.pth")


if __name__ == "__main__":
    train()
