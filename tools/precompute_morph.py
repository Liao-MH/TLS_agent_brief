#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Precompute morphology feature arrays (.npy) for tiles and save them alongside images.

Default mapping (must match projects/morph_inputfusion/transforms.py):
  <img_root>/.../images/.../<name>.<ext>
-><morph_root>/.../morph/.../<name>.npy

Channel order MUST match projects/morph_inputfusion/channel_spec.py:
  0: H          (Hematoxylin concentration from H&E color deconvolution; fixed clip to [0,1])
  1: GradMag
  2: LBP
  3: Gabor_S1
  4: Gabor_S2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable
from tqdm import tqdm

import numpy as np

from projects.morph_inputfusion.morph_features import MorphComputeCfg, compute_morph_array

# Prefer OpenCV (typically already available in mmseg environments via mmcv deps).
try:
    import cv2  # type: ignore
except Exception as e:
    raise RuntimeError(
        "OpenCV (cv2) is required for this script. "
        "Please ensure opencv-python is installed in your environment."
    ) from e


IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def compute_morph_array_from_path(rgb_path: Path, h_lo: float, h_hi: float) -> np.ndarray:
    """
    Return morphology array with shape [C, H, W], float32 in [0,1].
    Channel order must match projects/morph_inputfusion/channel_spec.py.
    """
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {rgb_path}")

    cfg = MorphComputeCfg(h_lo=float(h_lo), h_hi=float(h_hi))
    morph_hwc = compute_morph_array(bgr, cfg=cfg, input_color="bgr")
    morph = np.transpose(morph_hwc, (2, 0, 1)).astype(np.float32)
    return morph.clip(0.0, 1.0)


def iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--img-root", type=str, required=True,
        help="Root directory that contains the 'images' tree (or the images tree itself)."
    )
    ap.add_argument(
        "--morph-root", type=str, required=True,
        help="Root directory to write the 'morph' tree (parallel to images)."
    )
    ap.add_argument(
        "--img-dirname", type=str, default="images",
        help="Directory name used in path mapping (default: images)."
    )
    ap.add_argument(
        "--morph-dirname", type=str, default="morph",
        help="Directory name used in path mapping (default: morph)."
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing .npy files."
    )
    ap.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "float32"],
        help="Saved dtype for .npy (default: float16)."
    )

    # Fixed-clip parameters for H channel (dataset/global fixed)
    ap.add_argument(
        "--h-lo", type=float, default=0.0,
        help="Fixed clip lower bound for Hematoxylin concentration (default: 0.0)."
    )
    ap.add_argument(
        "--h-hi", type=float, default=1.5,
        help="Fixed clip upper bound for Hematoxylin concentration (default: 1.5). "
             "Adjust after inspecting Hc distribution; keep fixed across experiments."
    )

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    img_root = Path(args.img_root).resolve()
    morph_root = Path(args.morph_root).resolve()

    if not img_root.exists():
        raise RuntimeError(f"--img-root does not exist: {img_root}")

    morph_root.mkdir(parents=True, exist_ok=True)

    save_dtype = np.float16 if args.dtype == "float16" else np.float32

    n_total = 0
    n_done = 0
    n_skip = 0
    n_fail = 0

    # =========================
    # Percent progress without storing a giant list (extra memory ~0)
    # scans filesystem twice, but RAM is stable
    # =========================
    total = sum(1 for _ in iter_images(img_root))

    for img_path in tqdm(iter_images(img_root), total=total, desc="Images"):
        n_total += 1

        img_str = str(img_path)
        if f"{os.sep}{args.img_dirname}{os.sep}" in img_str:
            morph_str = img_str.replace(
                f"{os.sep}{args.img_dirname}{os.sep}",
                f"{os.sep}{args.morph_dirname}{os.sep}",
            )
            morph_path = Path(morph_str).with_suffix(".npy")
        else:
            rel = img_path.relative_to(img_root)
            morph_path = (morph_root / rel).with_suffix(".npy")

        morph_path.parent.mkdir(parents=True, exist_ok=True)

        if morph_path.exists() and not args.overwrite:
            n_skip += 1
            continue

        try:
            morph = compute_morph_array_from_path(img_path, h_lo=args.h_lo, h_hi=args.h_hi)
            morph = morph.astype(save_dtype, copy=False)
            np.save(str(morph_path), morph)
            n_done += 1
        except Exception as e:
            n_fail += 1
            print(f"[FAIL] {img_path} -> {morph_path}: {e}")

        if n_total % 200 == 0:
            print(f"[PROGRESS] scanned={n_total} done={n_done} skip={n_skip} fail={n_fail}")

    print(f"[SUMMARY] scanned={n_total} done={n_done} skip={n_skip} fail={n_fail}")
    print(f"[OUTPUT] morph tree root (expected by loader): {morph_root}")


if __name__ == "__main__":
    main()
