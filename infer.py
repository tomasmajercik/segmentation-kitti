import torch
from PIL import Image
import numpy as np
from models.unet import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet(n_classes=34).to(DEVICE)
model.load_state_dict(torch.load("unet_cityscape.pth"))
model.eval()

def infer(img_path):
    img = np.array(Image.open(img_path).convert("RGB"))
    tensor = torch.tensor(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        pred = model(tensor).argmax(1).squeeze().cpu().numpy()

    return pred

if __name__ == "__main__":
    infer("data/mini-cityscape/val/input/00000.png")