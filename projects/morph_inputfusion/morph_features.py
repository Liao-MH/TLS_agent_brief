# projects/morph_inputfusion/morph_features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Sequence

import numpy as np
import cv2


@dataclass
class MorphComputeCfg:
    h_lo: float = 0.0
    h_hi: float = 1.5
    sobel_ksize: int = 3
    lbp_enabled: bool = True

    gabor_gamma: float = 0.5
    gabor_orientations: Tuple[float, ...] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
    gabor_sigma_1: float = 2.0
    gabor_lambda_1: float = 6.0
    gabor_sigma_2: float = 4.0
    gabor_lambda_2: float = 12.0

    clip_lo: float = 0.0
    clip_hi: float = 1.0


CHANNEL_ORDER_DEFAULT = ("H", "GradMag", "LBP", "Gabor_S1", "Gabor_S2")


def _minmax01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + eps)


def _bgr_u8_to_rgb01(bgr_u8: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _rgb2od(rgb01: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rgb01 = rgb01.astype(np.float32, copy=False)
    rgb01 = np.clip(rgb01, eps, 1.0)
    return (-np.log(rgb01)).astype(np.float32)


def _color_deconv_he(rgb01: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    he_matrix = np.array(
        [
            [0.650, 0.072],
            [0.704, 0.990],
            [0.286, 0.105],
        ],
        dtype=np.float32,
    )
    he_matrix /= np.linalg.norm(he_matrix, axis=0, keepdims=True) + 1e-8

    od = _rgb2od(rgb01)
    od_flat = od.reshape(-1, 3).T

    C, *_ = np.linalg.lstsq(he_matrix, od_flat, rcond=None)
    C = C.T

    Hc = C[:, 0].reshape(rgb01.shape[:2]).astype(np.float32)
    Ec = C[:, 1].reshape(rgb01.shape[:2]).astype(np.float32)

    recon = (he_matrix @ C.T).T.reshape(od.shape).astype(np.float32)
    residual = np.maximum(od - recon, 0.0).mean(axis=2).astype(np.float32)

    Hc = np.maximum(Hc, 0.0)
    Ec = np.maximum(Ec, 0.0)
    residual = np.maximum(residual, 0.0)
    return Hc, Ec, residual


def _h_channel_01(rgb01: np.ndarray, h_lo: float, h_hi: float) -> np.ndarray:
    Hc, _, _ = _color_deconv_he(rgb01)
    denom = float(h_hi - h_lo)
    if denom <= 1e-12:
        return np.zeros_like(Hc, dtype=np.float32)
    out = (Hc - float(h_lo)) / denom
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def _gradmag_01(rgb01: np.ndarray, ksize: int = 3) -> np.ndarray:
    gray = cv2.cvtColor((rgb01 * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=int(ksize))
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=int(ksize))
    mag = np.sqrt(gx * gx + gy * gy)
    return _minmax01(mag)


def _lbp_01(rgb01: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor((rgb01 * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape[:2]
    out = np.zeros_like(gray, dtype=np.float32)
    if h < 3 or w < 3:
        return out

    center = gray[1:h - 1, 1:w - 1]
    lbp = np.zeros_like(center, dtype=np.uint8)

    neighbors = [
        (-1, -1, 7),
        (-1,  0, 6),
        (-1,  1, 5),
        ( 0,  1, 4),
        ( 1,  1, 3),
        ( 1,  0, 2),
        ( 1, -1, 1),
        ( 0, -1, 0),
    ]

    for dy, dx, bit in neighbors:
        y0 = 1 + dy
        y1 = (h - 1) + dy
        x0 = 1 + dx
        x1 = (w - 1) + dx
        nb = gray[y0:y1, x0:x1]
        lbp |= ((nb >= center).astype(np.uint8) << bit)

    out[1:h - 1, 1:w - 1] = lbp.astype(np.float32) / 255.0
    return out.clip(0.0, 1.0)


def _gabor_bank_response(gray_u8: np.ndarray, sigma: float, lambd: float,
                         gamma: float, orientations: Sequence[float]) -> np.ndarray:
    acc = None
    gray_f = gray_u8.astype(np.float32)

    for theta in orientations:
        ksize = int(max(7, round(float(sigma) * 6)))
        if ksize % 2 == 0:
            ksize += 1
        kernel = cv2.getGaborKernel(
            (ksize, ksize),
            sigma=float(sigma),
            theta=float(theta),
            lambd=float(lambd),
            gamma=float(gamma),
            psi=0,
            ktype=cv2.CV_32F,
        )
        resp = cv2.filter2D(gray_f, cv2.CV_32F, kernel)
        resp = np.abs(resp)
        acc = resp if acc is None else np.maximum(acc, resp)

    if acc is None:
        acc = np.zeros_like(gray_f, dtype=np.float32)
    return _minmax01(acc)


def compute_morph_hwc_from_bgr(
    bgr: np.ndarray,
    cfg: Optional[MorphComputeCfg] = None,
    channel_order: Sequence[str] = CHANNEL_ORDER_DEFAULT,
) -> np.ndarray:
    cfg = cfg or MorphComputeCfg()
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] < 3:
        raise ValueError(f"Invalid BGR image shape={None if bgr is None else bgr.shape}")

    bgr_u8 = bgr if bgr.dtype == np.uint8 else np.clip(bgr, 0, 255).astype(np.uint8)
    rgb01 = _bgr_u8_to_rgb01(bgr_u8)

    H = _h_channel_01(rgb01, h_lo=cfg.h_lo, h_hi=cfg.h_hi)
    Gm = _gradmag_01(rgb01, ksize=cfg.sobel_ksize)
    L = _lbp_01(rgb01) if cfg.lbp_enabled else np.zeros_like(H, dtype=np.float32)

    gray = cv2.cvtColor((rgb01 * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    Gb1 = _gabor_bank_response(gray, cfg.gabor_sigma_1, cfg.gabor_lambda_1, cfg.gabor_gamma, cfg.gabor_orientations)
    Gb2 = _gabor_bank_response(gray, cfg.gabor_sigma_2, cfg.gabor_lambda_2, cfg.gabor_gamma, cfg.gabor_orientations)

    name2ch = {"H": H, "GradMag": Gm, "LBP": L, "Gabor_S1": Gb1, "Gabor_S2": Gb2}

    chans = []
    for nm in channel_order:
        if nm not in name2ch:
            raise KeyError(f"Unknown morph channel name '{nm}'. Known={list(name2ch.keys())}")
        chans.append(name2ch[nm])

    morph = np.stack(chans, axis=-1).astype(np.float32)
    morph = np.nan_to_num(morph, nan=0.0, posinf=0.0, neginf=0.0)
    morph = np.clip(morph, float(cfg.clip_lo), float(cfg.clip_hi))
    return morph


def compute_morph_array(
    img: np.ndarray,
    cfg: Optional[MorphComputeCfg] = None,
    channel_order: Sequence[str] = CHANNEL_ORDER_DEFAULT,
    input_color: str = "bgr",
) -> np.ndarray:
    """
    Compute morphology feature array from RGB/BGR image.

    Args:
        img: Input image (H,W,3) in uint8 or float32 (0-1 or 0-255).
        cfg: Morphology compute config.
        channel_order: Output channel order.
        input_color: "bgr" or "rgb".

    Returns:
        np.float32 array with shape (H,W,C) and range [0,1].
    """
    if img is None or img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"Invalid image shape={None if img is None else img.shape}")
    if input_color not in {"bgr", "rgb"}:
        raise ValueError(f"input_color must be 'bgr' or 'rgb', got {input_color!r}")

    if img.dtype == np.uint8:
        img_u8 = img
    else:
        max_val = float(np.nanmax(img))
        if max_val <= 1.0:
            img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)

    bgr = img_u8 if input_color == "bgr" else cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    return compute_morph_hwc_from_bgr(bgr=bgr, cfg=cfg, channel_order=channel_order)
