from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.preprocessing import build_transforms


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    top_k: List[Dict[str, float]]


def _apply_jet_colormap(gray_u8: np.ndarray) -> np.ndarray:
    """Approximate OpenCV COLORMAP_JET for a uint8 grayscale image.

    Returns RGB uint8 array (H, W, 3).
    """

    if gray_u8.dtype != np.uint8:
        gray_u8 = gray_u8.astype(np.uint8)

    x = gray_u8.astype(np.float32) / 255.0

    # Piecewise-linear "jet" approximation
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0 + 0.5).astype(np.uint8)


def load_torch_model(pth_path: str | Path, *, device: str) -> torch.nn.Module:
    p = Path(pth_path)
    model = torch.load(p, weights_only=False, map_location=device)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image: Image.Image,
    class_labels: List[str],
    *,
    image_size: int = 224,
    top_k: int = 5,
    device: str,
) -> Tuple[str, float, List[Dict[str, float]]]:
    _, val_t = build_transforms(image_size=image_size)

    if image.mode != "RGB":
        image = image.convert("RGB")

    x = val_t(image).unsqueeze(0).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0)

    k = min(int(top_k), probs.numel())
    confs, idxs = torch.topk(probs, k=k)

    top = []
    for conf, idx in zip(confs.detach().cpu().tolist(), idxs.detach().cpu().tolist()):
        label = class_labels[int(idx)] if int(idx) < len(class_labels) else f"class_{int(idx)}"
        top.append({"label": label, "confidence": float(conf)})

    best = top[0]
    return best["label"], float(best["confidence"]), top


@torch.no_grad()
def predict_images(
    model: torch.nn.Module,
    images: List[Image.Image],
    class_labels: List[str],
    *,
    image_size: int = 224,
    top_k: int = 5,
    device: str,
) -> List[Tuple[str, float, List[Dict[str, float]]]]:
    """Batch predict a list of PIL images.

    Returns one (label, confidence, top_k_list) per image.
    """

    if not images:
        return []

    _, val_t = build_transforms(image_size=image_size)

    xs: List[torch.Tensor] = []
    for image in images:
        if image.mode != "RGB":
            image = image.convert("RGB")
        xs.append(val_t(image))

    x = torch.stack(xs, dim=0).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1)

    k = min(int(top_k), int(probs.shape[1]))
    confs, idxs = torch.topk(probs, k=k, dim=1)

    out: List[Tuple[str, float, List[Dict[str, float]]]] = []
    for row_confs, row_idxs in zip(confs.detach().cpu(), idxs.detach().cpu()):
        top: List[Dict[str, float]] = []
        for conf, idx in zip(row_confs.tolist(), row_idxs.tolist()):
            label = class_labels[int(idx)] if int(idx) < len(class_labels) else f"class_{int(idx)}"
            top.append({"label": label, "confidence": float(conf)})
        best = top[0]
        out.append((best["label"], float(best["confidence"]), top))

    return out


def saliency_heatmap_png(
    model: torch.nn.Module,
    image: Image.Image,
    class_labels: List[str],
    *,
    image_size: int = 224,
    device: str,
    target_index: int | None = None,
) -> Tuple[str, int, bytes]:
    """Compute a simple gradient saliency heatmap.

    This is architecture-agnostic and works with any classifier that maps
    NCHW -> logits.

    Returns: (target_label, target_index, heatmap_png_bytes)
    """

    _, val_t = build_transforms(image_size=image_size)

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Prepare input for gradient-based attribution
    x = val_t(image).unsqueeze(0).to(device)
    x.requires_grad_(True)

    # Forward
    model.zero_grad(set_to_none=True)
    logits = model(x)

    if target_index is None:
        target_index = int(torch.argmax(logits, dim=1).item())

    # Backward on target logit
    score = logits[0, int(target_index)]
    score.backward()

    if x.grad is None:
        raise RuntimeError("failed to compute gradients for saliency")

    # Saliency: max abs gradient across channels -> HxW
    sal = x.grad.detach()[0].abs().max(dim=0)[0]
    sal = sal - sal.min()
    sal = sal / (sal.max() + 1e-8)

    sal_u8 = (sal.detach().cpu().numpy() * 255.0).astype(np.uint8)

    # Resize heatmap to original image size
    w, h = image.size

    if sal_u8.shape[0] != h or sal_u8.shape[1] != w:
        sal_u8 = np.array(Image.fromarray(sal_u8).resize((w, h), resample=Image.Resampling.BICUBIC))

    # Colorize (RGB)
    heat_rgb = _apply_jet_colormap(sal_u8)

    out = io.BytesIO()
    Image.fromarray(heat_rgb).save(out, format="PNG")
    heatmap_png = out.getvalue()

    label = class_labels[int(target_index)] if int(target_index) < len(class_labels) else f"class_{int(target_index)}"
    return label, int(target_index), heatmap_png
