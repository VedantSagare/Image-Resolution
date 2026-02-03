import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from models.espcn import ESPCN
from utils.dataset import SRDataset
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr-dir", default="data/train/LR")
    parser.add_argument("--hr-dir", default="data/train/HR")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    dataset = SRDataset(args.lr_dir, args.hr_dir)
    use_cuda = device == "cuda"
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2,
    )

    model = ESPCN(scale_factor=args.scale).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and use_cuda)

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for lr, hr in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=args.amp and use_cuda):
                sr = model(lr)
                loss = criterion(sr, hr)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
        torch.save(model.state_dict(), f"checkpoints/espcn_epoch{epoch+1}.pth")

if __name__ == "__main__":
    main()
