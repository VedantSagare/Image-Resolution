import argparse
import os
import torch
from models.espcn import ESPCN
from utils.image_utils import load_image, save_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input.jpg")
    parser.add_argument("--output", default="outputs/output.png")
    parser.add_argument("--checkpoint", default="checkpoints/espcn_epoch10.pth")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda = device == "cuda"

    model = ESPCN(scale_factor=args.scale).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    lr = load_image(args.input).to(device)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=args.amp and use_cuda):
            sr = model(lr)

    save_image(sr, args.output)

if __name__ == "__main__":
    main()
