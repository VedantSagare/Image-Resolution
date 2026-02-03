import os
import cv2
import torch
from torch.utils.data import Dataset

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        lr_files = [
            f for f in os.listdir(lr_dir)
            if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
        ]
        lr_files.sort()
        self.pairs = [
            (f, f) for f in lr_files
            if os.path.exists(os.path.join(hr_dir, f))
        ]
        if not self.pairs:
            raise FileNotFoundError(
                f"No matching image pairs found in {lr_dir} and {hr_dir}"
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr_name, hr_name = self.pairs[idx]
        lr = cv2.imread(os.path.join(self.lr_dir, lr_name), cv2.IMREAD_COLOR)
        hr = cv2.imread(os.path.join(self.hr_dir, hr_name), cv2.IMREAD_COLOR)
        if lr is None or hr is None:
            raise FileNotFoundError(f"Failed to read pair: {lr_name}, {hr_name}")

        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)

        lr = torch.from_numpy(lr).permute(2, 0, 1).contiguous().float() / 255.0
        hr = torch.from_numpy(hr).permute(2, 0, 1).contiguous().float() / 255.0

        return lr, hr
