import torch
import numpy as np
from PIL import Image
from models.unet import UNet
# from models.unet_mini import UNet
import matplotlib.pyplot as plt

"""
Inference script for semantic segmentation using a pre-trained UNet model.
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet(n_classes=32).to(DEVICE)
model.load_state_dict(torch.load("models/weights/unet_weights_best.pth", map_location=torch.device(DEVICE)))
model.eval()

# Function to perform inference on a single image
def infer(img_path):
    img = np.array(Image.open(img_path).convert("RGB"))
    tensor = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = tensor.to(DEVICE)

    with torch.no_grad(): # disable gradient computation for inference
        pred = model(tensor).argmax(1).squeeze().cpu().numpy()

    return img, pred

# Function to plot original image and predicted segmentation
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
    img, pred = infer("data/mini-cityscape/val/input/00003.png")
    plot_segmentation(img, pred)