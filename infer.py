import torch
from models.espcn import ESPCN
from utils.image_utils import load_image, save_image

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ESPCN(scale_factor=2).to(device)
model.load_state_dict(torch.load("checkpoints/espcn_epoch10.pth"))
model.eval()

lr = load_image("input.jpg").to(device)

with torch.no_grad():
    sr = model(lr)

save_image(sr, "outputs/output.png")
