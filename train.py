import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
from models.unet import UNet
from dataset.mini_cityscape_dataset import MiniCityscapeDataset

devide = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {devide}")

def train(batch_size=8, n_classes=32, lr=1e-4, epochs=10):
    wandb.init(
    project="cityscape_unet_training",
    config={
        "batch_size": batch_size,
        "n_classes": n_classes,
        "learning_rate": lr,
        "epochs": epochs
    }
)
    """
    train_dataset = MiniCityscapeDataset(root="data/mini-cityscape", split='train')
    val_dataset = MiniCityscapeDataset(root="data/mini-cityscape", split='val')
    """
    train_full = MiniCityscapeDataset(root="data/mini-cityscape", split='train')
    val_full = MiniCityscapeDataset(root="data/mini-cityscape", split='val')

    # take only first 2 samples for training
    train_dataset = torch.utils.data.Subset(train_full, [0, 1])

    # take only first 1 sample for validation
    val_dataset = torch.utils.data.Subset(val_full, [0])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = UNet(n_classes=n_classes).to(devide)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    #wandb.watch(model, log="all", log_freq=100)

    for epoch in range(epochs):
        model.train()

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(devide), masks.to(devide)

            predictions = model(imgs)
            loss = F.cross_entropy(predictions, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        wandb.log({"epoch": epoch+1, "train_loss": loss.item()})

        #simple validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                predictions = model(imgs.to(devide))
                total_val_loss += F.cross_entropy(predictions, masks.to(devide)).item()

        print(f"Validation Loss: {total_val_loss/len(val_loader):.4f}")

        wandb.log({
                "example_image": wandb.Image(imgs[0].cpu(), caption="Input"),
            "example_mask": wandb.Image(masks[0].cpu(), caption="Ground truth"),
            "example_pred": wandb.Image(predictions.argmax(dim=1)[0].cpu(), caption="Prediction")
        })


    torch.save(model.state_dict(), "unet_weights.pth")
    print("Model weights saved to unet_weights.pth")

if __name__ == "__main__":
    train()