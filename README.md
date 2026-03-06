# Generative AI for Super-Resolution of Satellite Micro-Regions

Project scaffold for GAN-based super-resolution using PyTorch and a Streamlit frontend.

Files:
- app.py — Streamlit app to upload images and display SR results.
- model.py — Generator (RRDB), Discriminator (VGG-style), losses, and training loop skeleton.
- utils.py — image IO, bicubic upsampling, PSNR/SSIM metrics.
- requirements.txt — Python dependencies.

Notes:
- Place trained generator weights at `sr_model.pth` in the project root for inference.
- The `train` function in `model.py` is a skeleton; adapt dataloaders and hyperparameters for production training.
