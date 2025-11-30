import torch
import numpy as np
from PIL import Image
from models.unet import UNet
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet(n_classes=32).to(DEVICE)
model.load_state_dict(torch.load("unet_weights3.pth", map_location=torch.device(DEVICE)))
model.eval()

def infer(img_path):
    img = np.array(Image.open(img_path).convert("RGB"))
    tensor = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        pred = model(tensor).argmax(1).squeeze().cpu().numpy()

    return img, pred

def plot_segmentation(img, pred):
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.title("Predicted Segmentation")
    plt.imshow(pred, cmap='tab20')  # or any colormap
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    img, pred = infer("data/mini-cityscape/val/input/00000.png")
    plot_segmentation(img, pred)