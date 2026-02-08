

"""Random overlay check for RGB vs morphology channel alignment."""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from .channel_spec import CHANNEL_SPECS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', required=True)
    ap.add_argument('--split', default='train_gc')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--img-dir', default='images')
    ap.add_argument('--morph-dir', default='morph')
    ap.add_argument('--morph-ext', default='.npy')
    ap.add_argument('--morph-channel', default='GradMag')
    ap.add_argument('--seed', type=int, default=123)
    args = ap.parse_args()

    random.seed(args.seed)

    split_dir = Path(args.data_root) / args.split
    img_dir = split_dir / args.img_dir
    morph_dir = split_dir / args.morph_dir
    out_dir = Path('_morph_sanity_out')
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in img_dir.glob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}])
    if not imgs:
        raise RuntimeError(f'No images found in: {img_dir}')

    pick = random.sample(imgs, k=min(args.n, len(imgs)))
    ch_idx = CHANNEL_SPECS[args.morph_channel]

    for p in pick:
        rgb = cv2.imread(str(p), cv2.IMREAD_COLOR)  # BGR uint8
        morph_path = morph_dir / (p.stem + args.morph_ext)
        arr = np.load(str(morph_path))
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[0] < arr.shape[-1]:
            arr = np.transpose(arr, (1, 2, 0))
        m = arr[:, :, ch_idx].astype(np.float32)
        m = np.clip(m, 0, 1)

        heat = (m * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(rgb, 0.7, heat, 0.3, 0)

        cv2.imwrite(str(out_dir / f'{p.stem}__{args.morph_channel}_overlay.png'), overlay)

    print(f'[OK] overlays -> {out_dir.resolve()}')


if __name__ == '__main__':
    main()
