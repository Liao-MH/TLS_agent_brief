from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np

from .utils import ensure_dir, write_json
from .constants import SCHEMA_VERSION


@dataclass
class PatchInfo:
    idx: int
    level: int
    x0: int
    y0: int
    x1: int
    y1: int
    cx0: int
    cy0: int
    cx1: int
    cy1: int
    tissue_fraction: float


def tissue_fraction_for_patch(
    x0_l: int, y0_l: int, x1_l: int, y1_l: int,
    tissue_mask_thumb: np.ndarray,
    base_to_thumb_ds: float,
    level_downsample: float,
) -> float:
    """
    Map level-coordinates -> base -> thumb mask coordinates to compute tissue fraction.
    """
    bx0 = float(x0_l) * level_downsample
    by0 = float(y0_l) * level_downsample
    bx1 = float(x1_l) * level_downsample
    by1 = float(y1_l) * level_downsample

    h_m, w_m = tissue_mask_thumb.shape
    mx0 = max(int(bx0 / base_to_thumb_ds), 0)
    my0 = max(int(by0 / base_to_thumb_ds), 0)
    mx1 = min(int(np.ceil(bx1 / base_to_thumb_ds)), w_m)
    my1 = min(int(np.ceil(by1 / base_to_thumb_ds)), h_m)
    if mx1 <= mx0 or my1 <= my0:
        return 0.0
    patch_mask = tissue_mask_thumb[my0:my1, mx0:mx1]
    return float(patch_mask.mean())


def plan_patches(
    level_dim: Tuple[int, int],
    level: int,
    patch_size: int,
    overlap: int,
    tissue_mask_thumb: np.ndarray,
    base_to_thumb_ds: float,
    level_downsample: float,
    tissue_thresh: float,
    out_dir: str,
) -> List[PatchInfo]:
    """
    Generate patch list on a given pyramid level.
    """
    out_dir = ensure_dir(out_dir)
    analysis_dir = ensure_dir(Path(out_dir) / "artifacts" / "analysis")

    W, H = level_dim
    patch_size = int(patch_size)
    overlap = int(overlap)
    stride = patch_size - overlap
    margin = overlap // 2

    patches: List[PatchInfo] = []
    idx = 0
    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            x1 = min(x0 + patch_size, W)
            y1 = min(y0 + patch_size, H)
            if x1 <= x0 or y1 <= y0:
                continue

            frac = tissue_fraction_for_patch(
                x0, y0, x1, y1,
                tissue_mask_thumb=tissue_mask_thumb,
                base_to_thumb_ds=base_to_thumb_ds,
                level_downsample=level_downsample,
            )
            if frac < float(tissue_thresh):
                continue

            cx0 = max(x0 + margin, x0)
            cy0 = max(y0 + margin, y0)
            cx1 = min(x0 + patch_size - margin, x1)
            cy1 = min(y0 + patch_size - margin, y1)

            patches.append(PatchInfo(
                idx=idx, level=level,
                x0=x0, y0=y0, x1=x1, y1=y1,
                cx0=cx0, cy0=cy0, cx1=cx1, cy1=cy1,
                tissue_fraction=frac,
            ))
            idx += 1

    patch_plan = {
        "schema_version": SCHEMA_VERSION,
        "level": level,
        "level_dim": {"width": W, "height": H},
        "patch_size": patch_size,
        "overlap": overlap,
        "stride": stride,
        "margin": margin,
        "tissue_thresh": float(tissue_thresh),
        "num_patches": len(patches),
        "patches": [asdict(p) for p in patches],
    }
    write_json(str(Path(analysis_dir) / "patch_plan.json"), patch_plan)

    return patches
