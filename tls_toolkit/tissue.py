from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np
import cv2

from .utils import ensure_dir
from .constants import SCHEMA_VERSION


@dataclass
class TissueResult:
    tissue_mask: np.ndarray      # uint8 0/1
    base_to_mask_ds: float
    thumb_path: str
    tissue_mask_path: str
    thumb_overlay_path: str
    qc: Dict[str, Any]


def _connected_components_stats(mask01: np.ndarray) -> Dict[str, Any]:
    mask = (mask01 > 0).astype(np.uint8)
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if num > 1 else np.array([], dtype=np.int64)
    total = int(mask.sum())
    largest = int(areas.max()) if areas.size > 0 else 0
    return {
        "components": int(max(num - 1, 0)),
        "total_area_px": total,
        "largest_component_area_px": largest,
        "largest_component_fraction": float(largest / total) if total > 0 else 0.0,
    }


def run_tissue(
    tissue_mask: np.ndarray,
    thumb_rgb: np.ndarray,
    out_dir: str,
) -> TissueResult:
    """
    Save tissue artifacts + compute basic QC.
    """
    out_dir = ensure_dir(out_dir)
    qc_dir = ensure_dir(Path(out_dir) / "artifacts" / "qc")

    tissue_mask_path = str(Path(qc_dir) / "tissue_mask.png")
    thumb_path = str(Path(qc_dir) / "thumb.png")
    thumb_overlay_path = str(Path(qc_dir) / "thumb_overlay.png")

    cv2.imwrite(tissue_mask_path, (tissue_mask * 255).astype(np.uint8))
    cv2.imwrite(thumb_path, cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR))

    overlay = thumb_rgb.copy()
    contours, _ = cv2.findContours((tissue_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
    cv2.imwrite(thumb_overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    frac = float(tissue_mask.mean())
    cc = _connected_components_stats(tissue_mask)

    qc = {
        "schema_version": SCHEMA_VERSION,
        "tissue_fraction_thumb": frac,
        "tissue_components": cc["components"],
        "tissue_largest_component_fraction": cc["largest_component_fraction"],
    }

    return TissueResult(
        tissue_mask=tissue_mask,
        base_to_mask_ds=0.0,  # caller may fill
        thumb_path=thumb_path,
        tissue_mask_path=tissue_mask_path,
        thumb_overlay_path=thumb_overlay_path,
        qc=qc,
    )
