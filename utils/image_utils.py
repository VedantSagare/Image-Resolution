import os
import cv2
import torch

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float() / 255.0
    return img.unsqueeze(0)

def save_image(tensor, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img = tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = (img * 255.0).round().clip(0, 255).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)
