# Image Super-Resolution Using CNN (PyTorch)

This project enhances low-resolution images into high-resolution images using
modern CNN-based Super-Resolution models such as ESPCN and EDSR.

## Features
- ESPCN & EDSR models
- PyTorch implementation
- PSNR & SSIM metrics
- Train & inference pipelines
- Clean modular structure

## Installation
pip install -r requirements.txt

## Dataset Structure
data/train/LR
data/train/HR

## Training
python train.py

## Inference
python infer.py

## Output
Enhanced images are saved in outputs/

## License
MIT
