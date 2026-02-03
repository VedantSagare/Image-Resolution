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
python train.py --epochs 10 --batch-size 8 --num-workers 2 --amp

## Inference
python infer.py --input input.jpg --output outputs/output.png --amp

## Output
Enhanced images are saved in outputs/

## License
MIT
