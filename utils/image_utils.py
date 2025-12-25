import cv2
import torch

def load_image(path):
    img = cv2.imread(path)
    img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
    return img.unsqueeze(0)

def save_image(tensor, path):
    img = tensor.squeeze().permute(1,2,0).cpu().numpy() * 255
    cv2.imwrite(path, img)
