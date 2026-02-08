# tools/morph_sanity_check.py

from __future__ import annotations

import argparse
import os
import os.path as osp
import random
from typing import List, Tuple

import cv2
import numpy as np

from projects.morph_fusion.morph_channels import resolve_morph_channel_indices


def list_images(img_dir: str, exts: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')) -> List[str]:
    paths = []
    for root, _, files in os.walk(img_dir):
        for f in files:
            if osp.splitext(f)[1].lower() in exts:
                paths.append(osp.join(root, f))
    return sorted(paths)


def infer_morph_path(img_path: str, morph_dir: str | None, suffix: str, ext: str) -> str:
    stem = osp.splitext(osp.basename(img_path))[0]
    if morph_dir is not None:
        return osp.join(morph_dir, stem + ext)
    return osp.join(osp.dirname(img_path), stem + suffix + ext)


def load_morph(morph_path: str) -> np.ndarray:
    if morph_path.endswith('.npz'):
        npz = np.load(morph_path)
        keys = sorted(list(npz.keys()))
        if not keys:
            raise ValueError(f'Empty npz: {morph_path}')
        arr = npz[keys[0]]
    else:
        arr = np.load(morph_path, allow_pickle=False)

    if arr.ndim == 2:
        arr = arr[:, :, None]
    elif arr.ndim == 3:
        # allow CHW
        if arr.shape[0] < 16 and arr.shape[0] != arr.shape[1] and arr.shape[0] != arr.shape[2]:
            arr = np.transpose(arr, (1, 2, 0))
    else:
        raise ValueError(f'Unsupported morph array shape: {arr.shape}')
    return arr.astype(np.float32, copy=False)


def overlay_heatmap(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    rgb: uint8 HWC, RGB
    heat: float32 HW in [0,1] or arbitrary -> will normalize
    """
    h = heat.copy()
    h = h - np.min(h)
    denom = np.max(h) + 1e-6
    h = (h / denom * 255.0).astype(np.uint8)

    cm = cv2.applyColorMap(h, cv2.COLORMAP_JET)  # BGR
    cm = cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)

    out = (rgb.astype(np.float32) * (1 - alpha) + cm.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--img-dir', required=True, help='Directory containing RGB tiles')
    ap.add_argument('--out-dir', required=True, help='Output directory for overlays')
    ap.add_argument('--morph-dir', default=None, help='Directory containing morph files (optional)')
    ap.add_argument('--morph-suffix', default='_morph', help="Suffix used when morph_dir is None")
    ap.add_argument('--morph-ext', default='.npy', help="Morph file extension, e.g. .npy or .npz")
    ap.add_argument('--n', type=int, default=20, help='Number of random tiles to sample')
    ap.add_argument('--channel', default='GradMag', help='Morph channel name to visualize, e.g. GradMag/Gabor_S1')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    imgs = list_images(args.img_dir)
    if not imgs:
        raise RuntimeError(f'No images found under {args.img_dir}')

    random.seed(args.seed)
    sample = random.sample(imgs, k=min(args.n, len(imgs)))

    idx = resolve_morph_channel_indices([args.channel])[0]

    for i, img_path in enumerate(sample, 1):
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f'[WARN] Failed to read image: {img_path}')
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        morph_path = infer_morph_path(img_path, args.morph_dir, args.morph_suffix, args.morph_ext)
        if not osp.exists(morph_path):
            print(f'[WARN] Missing morph: {morph_path}')
            continue

        morph = load_morph(morph_path)
        if idx >= morph.shape[2]:
            print(f'[WARN] Channel index {idx} out of range for morph={morph.shape} @ {morph_path}')
            continue

        heat = morph[:, :, idx]
        out = overlay_heatmap(rgb, heat, alpha=0.45)

        stem = osp.splitext(osp.basename(img_path))[0]
        out_path = osp.join(args.out_dir, f'{stem}__overlay_{args.channel}.png')
        cv2.imwrite(out_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

        print(f'[{i}/{len(sample)}] saved {out_path}')


if __name__ == '__main__':
    main()

