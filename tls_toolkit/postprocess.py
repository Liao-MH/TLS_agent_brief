# tls_toolkit/postprocess.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import cv2

from .constants import POSTPROC_RULE_VERSION


@dataclass
class PostprocessResult:
    mask: np.ndarray
    stats: Dict[str, Any]


def _make_kernel(radius: int) -> np.ndarray:
    r = int(max(0, radius))
    k = 2 * r + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _remove_small_cc(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area in a binary mask."""
    min_area = int(max(0, min_area))
    if min_area <= 0:
        return mask_u8

    num, cc, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num <= 1:
        return mask_u8

    keep = np.zeros_like(mask_u8, dtype=np.uint8)
    for lbl in range(1, num):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area >= min_area:
            keep[cc == lbl] = 1
    return keep


def _smooth_binary(mask_u8: np.ndarray, close_r: int, open_r: int) -> np.ndarray:
    """Close then open to reduce jagged boundaries and spurs."""
    out = mask_u8
    if int(close_r) > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, _make_kernel(int(close_r)), iterations=1)
    if int(open_r) > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, _make_kernel(int(open_r)), iterations=1)
    return out


def _fill_external_contours(mask_u8: np.ndarray) -> np.ndarray:
    """
    Topology legalization at MASK level:
      - Extract external contours only
      - Re-fill them to eliminate holes and weird internal artifacts
    This makes downstream polygonization more stable and QuPath-friendly.
    """
    if int(mask_u8.sum()) == 0:
        return mask_u8
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(mask_u8, dtype=np.uint8)
    if contours:
        cv2.drawContours(out, contours, contourIdx=-1, color=1, thickness=-1)
    return out


def _sanitize_binary(
    mask_u8: np.ndarray,
    *,
    min_area: int,
    close_r: int,
    open_r: int,
    fill_external: bool = True,
) -> np.ndarray:
    """
    Unified mask-level geometry legalization:
      1) remove small CC
      2) smooth (close/open)
      3) fill external contours (remove holes; stabilize)
      4) remove small CC again (optional but useful after ops)
    """
    out = (mask_u8 > 0).astype(np.uint8)
    out = _remove_small_cc(out, min_area=min_area)
    out = _smooth_binary(out, close_r=close_r, open_r=open_r)
    if fill_external:
        out = _fill_external_contours(out)
    out = _remove_small_cc(out, min_area=min_area)
    return out


def postprocess_tls_gc_logic(
    full_mask: np.ndarray,
    tls_label: int,
    gc_label: int,
    immune_label: int,
    min_object_area: int = 50,
    *,
    # adjacency tolerance: TLS touches GC if dilated TLS overlaps GC
    touch_px: int = 1,
    # TLS definition: (TLS ∪ touched_GC) then dilate by expand_px
    expand_px: int = 5,
    # boundary smoothing at mask-level
    smooth_close_r: int = 2,
    smooth_open_r: int = 1,
) -> PostprocessResult:
    """
    NEW RULE (as requested):

    1) TLS component is TLS ONLY if it touches any GC (within touch_px).
       If NOT touching any GC => reclassify as ImmuneCluster.

    2) For touching TLS components, redefine TLS region:
         core = TLS_component ∪ touched_GC_pixels
         TLS_final = dilate(core, expand_px)

       This guarantees: TLS region *contains* GC region + 5px expansion.

    3) Geometry legalization happens at MASK level:
       - remove small objects (min_object_area)
       - close/open smoothing (smooth_close_r/open_r)
       - fill external contours (remove holes; stabilize polygons)

    Output mask keeps labels as:
      - GC stays gc_label
      - ImmuneCluster stays immune_label
      - TLS stays tls_label
    (But for vectorization we can define TLS geometry as TLS ∪ GC if needed.)
    """
    mask = full_mask.astype(np.int32, copy=True)

    tls0 = (mask == int(tls_label)).astype(np.uint8)
    gc0 = (mask == int(gc_label)).astype(np.uint8)
    immune0 = (mask == int(immune_label)).astype(np.uint8)

    # sanitize GC first (so "touch" is against cleaned GC)
    gc = _sanitize_binary(
        gc0,
        min_area=min_object_area,
        close_r=smooth_close_r,
        open_r=smooth_open_r,
        fill_external=True,
    )

    if int(tls0.sum()) == 0:
        # still sanitize immune for stability
        immune = _sanitize_binary(
            immune0,
            min_area=min_object_area,
            close_r=smooth_close_r,
            open_r=smooth_open_r,
            fill_external=True,
        )
        out = np.zeros_like(mask, dtype=np.uint8)
        out[immune > 0] = int(immune_label)
        out[gc > 0] = int(gc_label)
        return PostprocessResult(
            mask=out,
            stats=dict(
                rule_version=POSTPROC_RULE_VERSION,
                mode="tls_gc_logic",
                note="No TLS in input; only sanitized GC/Immune.",
                tls_components_total=0,
                tls_components_checked=0,
                tls_components_to_tls=0,
                tls_components_to_immune=0,
                touch_px=int(touch_px),
                expand_px=int(expand_px),
                min_object_area=int(min_object_area),
                smooth_close_r=int(smooth_close_r),
                smooth_open_r=int(smooth_open_r),
            ),
        )

    # find TLS connected components on raw TLS (we'll decide per-component)
    num, cc, stats_cc, _ = cv2.connectedComponentsWithStats(tls0, connectivity=8)

    touch_px = int(max(0, touch_px))
    expand_px = int(max(0, expand_px))
    touch_kernel = _make_kernel(touch_px) if touch_px > 0 else None
    expand_kernel = _make_kernel(expand_px) if expand_px > 0 else None

    tls_final = np.zeros_like(tls0, dtype=np.uint8)
    immune_add = np.zeros_like(tls0, dtype=np.uint8)

    checked = 0
    to_tls = 0
    to_immune = 0
    pixels_to_immune = 0
    pixels_tls_final = 0
    pixels_gc_included = 0

    H, W = tls0.shape

    for lbl in range(1, num):
        area = int(stats_cc[lbl, cv2.CC_STAT_AREA])
        if area < int(min_object_area):
            continue
        checked += 1

        x = int(stats_cc[lbl, cv2.CC_STAT_LEFT])
        y = int(stats_cc[lbl, cv2.CC_STAT_TOP])
        w = int(stats_cc[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats_cc[lbl, cv2.CC_STAT_HEIGHT])

        comp = (cc[y:y+h, x:x+w] == lbl).astype(np.uint8)
        if int(comp.sum()) == 0:
            continue

        # touch check against sanitized GC
        if touch_kernel is not None:
            comp_d = cv2.dilate(comp, touch_kernel, iterations=1)
        else:
            comp_d = comp

        gc_roi = gc[y:y+h, x:x+w]
        touches = bool((comp_d & gc_roi).any())

        if not touches:
            # whole component becomes immune
            comp_pixels = (cc == lbl)
            immune_add[comp_pixels] = 1
            to_immune += 1
            pixels_to_immune += int(comp_pixels.sum())
            continue

        # touches GC => TLS core = comp ∪ touched_GC, then expand by expand_px
        to_tls += 1
        touched_gc = (comp_d & gc_roi).astype(np.uint8)
        core = (comp | touched_gc).astype(np.uint8)
        pixels_gc_included += int(touched_gc.sum())

        if expand_kernel is None:
            tls_final[y:y+h, x:x+w] = np.maximum(tls_final[y:y+h, x:x+w], core)
            pixels_tls_final += int(core.sum())
        else:
            # pad ROI to avoid clipping during dilation
            pad = expand_px + 2
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(W, x + w + pad)
            y1 = min(H, y + h + pad)

            core_big = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            core_big[(y - y0):(y - y0 + h), (x - x0):(x - x0 + w)] = core
            expanded = cv2.dilate(core_big, expand_kernel, iterations=1)
            tls_final[y0:y1, x0:x1] = np.maximum(tls_final[y0:y1, x0:x1], expanded)
            pixels_tls_final += int(expanded.sum())

    # Immune = original immune + reclassified TLS
    immune = np.maximum((immune0 > 0).astype(np.uint8), immune_add)

    # sanitize TLS/Immune after construction to enforce stable polygons
    tls_final = _sanitize_binary(
        tls_final,
        min_area=min_object_area,
        close_r=smooth_close_r,
        open_r=smooth_open_r,
        fill_external=True,
    )
    immune = _sanitize_binary(
        immune,
        min_area=min_object_area,
        close_r=smooth_close_r,
        open_r=smooth_open_r,
        fill_external=True,
    )

    # IMPORTANT: enforce separation
    # - Immune must not overlap GC
    immune[gc > 0] = 0
    # - TLS is defined to contain GC, but we keep GC label separately in mask.
    #   So in label map, TLS should not overwrite GC pixels; we keep them as GC.
    tls_final[gc > 0] = 0
    # - Also prevent TLS/Immune overlap
    immune[tls_final > 0] = 0

    # compose final label map
    out = np.zeros((H, W), dtype=np.uint8)
    out[immune > 0] = int(immune_label)
    out[tls_final > 0] = int(tls_label)
    out[gc > 0] = int(gc_label)

    stats_out = {
        "rule_version": POSTPROC_RULE_VERSION,
        "mode": "tls_gc_logic",
        "touch_px": int(touch_px),
        "expand_px": int(expand_px),
        "min_object_area": int(min_object_area),
        "smooth_close_r": int(smooth_close_r),
        "smooth_open_r": int(smooth_open_r),

        "tls_components_total": int(max(num - 1, 0)),
        "tls_components_checked": int(checked),
        "tls_components_to_tls": int(to_tls),
        "tls_components_to_immune": int(to_immune),
        "tls_pixels_to_immune": int(pixels_to_immune),

        "tls_pixels_final": int(tls_final.sum()),
        "gc_pixels_sanitized": int(gc.sum()),
        "immune_pixels_final": int(immune.sum()),
        "gc_pixels_included_in_tls_core_local": int(pixels_gc_included),

        "note": "TLS kept only if touches GC; TLS region redefined as (TLS∪touched_GC) then dilate; mask-level sanitize+smooth; label map keeps GC separate.",
    }
    return PostprocessResult(mask=out, stats=stats_out)
