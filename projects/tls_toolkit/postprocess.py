from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
from tqdm import tqdm

import numpy as np
import cv2

from .constants import POSTPROC_RULE_VERSION


@dataclass
class PostprocessResult:
    mask: np.ndarray
    stats: Dict[str, Any]


def postprocess_tls_gc_proximity(
    full_mask: np.ndarray,
    tls_label: int,
    gc_label: int,
    immune_label: int,
    min_object_area: int,
    proximity_px: int,
) -> PostprocessResult:
    """
    Rule (versioned): TLS components that are NOT within proximity of GC are converted to ImmuneCluster.

    Proximity definition:
      - For each TLS CC (above min_object_area), dilate it by proximity_px and
        check if it intersects GC mask.
    """
    mask = full_mask.copy()

    tls_mask = (mask == tls_label).astype(np.uint8)
    gc_mask = (mask == gc_label).astype(np.uint8)

    num_labels, tls_cc, stats, _ = cv2.connectedComponentsWithStats(tls_mask, connectivity=8)

    converted_components = 0
    converted_pixels = 0
    checked_components = 0

    gc_mask_u8 = (gc_mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * proximity_px + 1, 2 * proximity_px + 1))

    for lbl in tqdm(
        range(1, num_labels),
        total=max(0, num_labels - 1),
        desc="Postprocess TLS CC",
        dynamic_ncols=True,
    ):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < int(min_object_area):
            continue

        checked_components += 1

        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])

        comp = (tls_cc[y:y+h, x:x+w] == lbl).astype(np.uint8)
        if comp.sum() == 0:
            continue

        dil = cv2.dilate(comp, kernel, iterations=1)
        gc_reg = gc_mask_u8[y:y+h, x:x+w]

        has_gc = bool((dil & gc_reg).any())
        if not has_gc:
            comp_pixels = (tls_cc == lbl)
            npx = int(comp_pixels.sum())
            mask[comp_pixels] = int(immune_label)
            converted_components += 1
            converted_pixels += npx

    stats_out = {
        "rule_version": POSTPROC_RULE_VERSION,
        "tls_components_total": int(max(num_labels - 1, 0)),
        "tls_components_checked": int(checked_components),
        "tls_components_converted": int(converted_components),
        "tls_pixels_converted": int(converted_pixels),
        "proximity_px": int(proximity_px),
        "min_object_area": int(min_object_area),
    }
    return PostprocessResult(mask=mask, stats=stats_out)
