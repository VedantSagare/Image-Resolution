import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from models.espcn import ESPCN
from utils.dataset import SRDataset
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = SRDataset("data/train/LR", "data/train/HR")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = ESPCN(scale_factor=2).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    total_loss = 0

    for lr, hr in tqdm(loader):
        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr)
        loss = criterion(sr, hr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), f"checkpoints/espcn_epoch{epoch+1}.pth")
