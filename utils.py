import numpy as np
from PIL import Image
import torchvision.transforms as T
import torch
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def load_image_as_tensor(pil_img: Image.Image):
    # Return tensor in [0,1]
    transform = T.Compose([T.ToTensor()])
    return transform(pil_img)


def tensor_to_pil(tensor: torch.Tensor):
    tensor = tensor.detach().cpu().clamp(0, 1)
    transform = T.ToPILImage()
    return transform(tensor)


def bicubic_upscale_pil(pil_img: Image.Image, scale=4):
    w, h = pil_img.size
    return pil_img.resize((w * scale, h * scale), resample=Image.BICUBIC)


def pil_to_np(img: Image.Image):
    return np.array(img).astype(np.float32) / 255.0


def compute_psnr(pil_a: Image.Image, pil_b: Image.Image):
    a = pil_to_np(pil_a)
    b = pil_to_np(pil_b)
    # ensure same size
    if a.shape != b.shape:
        b = np.array(pil_b.resize(pil_a.size[::-1], resample=Image.BICUBIC)).astype(np.float32) / 255.0
    return compare_psnr(a, b, data_range=1.0)


def compute_ssim(pil_a: Image.Image, pil_b: Image.Image):
    a = pil_to_np(pil_a)
    b = pil_to_np(pil_b)
    if a.shape != b.shape:
        b = np.array(pil_b.resize(pil_a.size[::-1], resample=Image.BICUBIC)).astype(np.float32) / 255.0
    # compute ssim per-channel then average
    if a.ndim == 3:
        s = 0.0
        for c in range(a.shape[2]):
            s += compare_ssim(a[..., c], b[..., c], data_range=1.0)
        return s / a.shape[2]
    else:
        return compare_ssim(a, b, data_range=1.0)
