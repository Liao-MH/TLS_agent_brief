from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import tifffile

from .patches import PatchInfo
from .utils import ensure_dir, require_free_space
from .errors import ResourceError


@dataclass
class StitchMeta:
    mask_level: int
    width: int
    height: int
    dtype: str
    stitch_mode: str
    patch_size: int
    overlap: int


def stitch_center_crop(
    patches: List[PatchInfo],
    pred_by_idx: Dict[int, np.ndarray],
    level_dim: Tuple[int, int],
    out_dir: str,
    patch_size: int,
    overlap: int,
    mask_level: int,
) -> Tuple[str, StitchMeta]:
    """
    Deterministic stitch: write only center-crop region into full mask.
    Output mask is in mask_level coordinate system (same as patch coords).
    """
    out_dir = ensure_dir(out_dir)
    mask_dir = ensure_dir(Path(out_dir) / "artifacts" / "mask")
    full_mask_path = str(Path(mask_dir) / "full_mask.tif")

    W, H = level_dim
    ok, msg = require_free_space(mask_dir, need_bytes=int(H) * int(W) * 2)
    if not ok:
        raise ResourceError(f"Insufficient disk space for full mask: {msg}")

    memmap_path = full_mask_path + ".memmap"
    full_mm = np.memmap(memmap_path, dtype=np.uint8, mode="w+", shape=(H, W))
    full_mm[:] = 0

    for p in patches:
        pred = pred_by_idx.get(p.idx, None)
        if pred is None:
            continue

        py0 = p.cy0 - p.y0
        px0 = p.cx0 - p.x0
        py1 = p.cy1 - p.y0
        px1 = p.cx1 - p.x0

        Hp, Wp = pred.shape
        py0 = max(py0, 0); px0 = max(px0, 0)
        py1 = min(py1, Hp); px1 = min(px1, Wp)
        if py1 <= py0 or px1 <= px0:
            continue

        fy0, fx0, fy1, fx1 = p.cy0, p.cx0, p.cy1, p.cx1
        fy0 = max(fy0, 0); fx0 = max(fx0, 0)
        fy1 = min(fy1, H); fx1 = min(fx1, W)
        if fy1 <= fy0 or fx1 <= fx0:
            continue

        h_c = min(fy1 - fy0, py1 - py0)
        w_c = min(fx1 - fx0, px1 - px0)
        full_mm[fy0:fy0+h_c, fx0:fx0+w_c] = pred[py0:py0+h_c, px0:px0+w_c]

    full_arr = np.asarray(full_mm)
    tifffile.imwrite(full_mask_path, full_arr, dtype=np.uint8, bigtiff=True)

    del full_mm
    try:
        Path(memmap_path).unlink(missing_ok=True)
    except Exception:
        pass

    meta = StitchMeta(
        mask_level=int(mask_level),
        width=int(W),
        height=int(H),
        dtype="uint8",
        stitch_mode="center_crop",
        patch_size=int(patch_size),
        overlap=int(overlap),
    )
    return full_mask_path, meta
