import os
import cv2
import torch
from torch.utils.data import Dataset

class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.images = sorted(os.listdir(lr_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        lr = cv2.imread(os.path.join(self.lr_dir, self.images[idx]))
        hr = cv2.imread(os.path.join(self.hr_dir, self.images[idx]))

        lr = torch.from_numpy(lr).permute(2,0,1).float() / 255.0
        hr = torch.from_numpy(hr).permute(2,0,1).float() / 255.0

        return lr, hr
