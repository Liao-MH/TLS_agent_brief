
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .patches import PatchInfo
from .utils import ensure_dir


@dataclass
class QCTileSampleResult:
    qc_dir: str
    images_dir: str
    pred_dir: str
    merge_dir: str
    selected_counts: Dict[int, int]
    selected_tiles: List[int]


def _get_classes_palette(model: Any) -> Tuple[List[str], List[Tuple[int, int, int]]]:
    meta = getattr(model, "dataset_meta", None)
    if isinstance(meta, dict):
        classes = meta.get("classes")
        palette = meta.get("palette")
        if isinstance(classes, (list, tuple)) and isinstance(palette, (list, tuple)) and len(classes) == len(palette):
            pal = []
            for c in palette:
                if isinstance(c, (list, tuple)) and len(c) == 3:
                    pal.append((int(c[0]), int(c[1]), int(c[2])))
            if len(pal) == len(classes):
                return list(map(str, classes)), pal

    classes = ["background", "TLS", "GC", "ImmuneCluster"]
    palette = [
        (0, 0, 0),
        (0, 255, 0),
        (235, 143, 124),
        (0, 0, 255),
    ]
    return classes, palette


def _colorize_seg(pred: np.ndarray, palette: List[Tuple[int, int, int]]) -> np.ndarray:
    h, w = pred.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    max_id = len(palette) - 1
    pred_clip = np.clip(pred.astype(np.int32), 0, max_id)
    for cid, color in enumerate(palette):
        m = (pred_clip == cid)
        if m.any():
            out[m] = np.array(color, dtype=np.uint8)
    return out


def _save_merge_with_legend(
    img_rgb: np.ndarray,
    pred: np.ndarray,
    out_path: str,
    classes: List[str],
    palette: List[Tuple[int, int, int]],
    opacity: float = 0.5,
) -> None:
    seg_rgb = _colorize_seg(pred, palette)
    overlay = (seg_rgb.astype(np.float32) * (1.0 - opacity) + img_rgb.astype(np.float32) * opacity)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig = plt.figure(figsize=(14, 8))
    plt.imshow(overlay)
    plt.axis("off")

    patches = []
    for i, name in enumerate(classes):
        if i < len(palette):
            c = np.array(palette[i], dtype=np.float32) / 255.0
            patches.append(mpatches.Patch(color=c, label=name))
    if patches:
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, fontsize="large")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _compute_balanced_targets(total: int, avail: Dict[int, int], class_ids: Tuple[int, ...]) -> Dict[int, int]:
    if total <= 0:
        return {cid: 0 for cid in class_ids}
    base = total // len(class_ids)
    rem = total % len(class_ids)
    targets = {cid: base for cid in class_ids}

    order = sorted(class_ids, key=lambda c: avail.get(c, 0), reverse=True)
    for i in range(rem):
        targets[order[i % len(order)]] += 1

    deficit = 0
    for cid in class_ids:
        a = avail.get(cid, 0)
        if targets[cid] > a:
            deficit += targets[cid] - a
            targets[cid] = a

    if deficit > 0:
        for cid in order:
            cap = avail.get(cid, 0) - targets[cid]
            if cap <= 0:
                continue
            take = min(cap, deficit)
            targets[cid] += take
            deficit -= take
            if deficit == 0:
                break

    return targets


def save_qc_tile_samples_after_postprocess(
    *,
    wsi_backend: Any,
    model: Any,
    patches: List[PatchInfo],
    full_mask_level: np.ndarray,
    patch_size: int,
    level: int,
    out_dir: str,
    qc_tile_samples: int = 12,
    min_present_pixels: int = 200,
    opacity: float = 0.5,
    fg_ids: Tuple[int, ...] = (1, 2, 3),
) -> QCTileSampleResult:
    """
    Save QC tile samples based on FINAL mask (after postprocess).
    Output:
      artifacts/qc/tile_samples/{images,pred,merge}
    Sampling:
      - Try to pick N/3 per foreground class among tiles where that class is present in the FINAL mask.
      - If some class lacks candidates, redistribute.
    """
    out_dir = ensure_dir(out_dir)
    qc_dir = ensure_dir(Path(out_dir) / "artifacts" / "qc" / "tile_samples")
    images_dir = ensure_dir(Path(qc_dir) / "images")
    pred_dir = ensure_dir(Path(qc_dir) / "pred")
    merge_dir = ensure_dir(Path(qc_dir) / "merge")

    classes, palette = _get_classes_palette(model)

    # Build candidate lists per class based on FINAL mask presence
    candidates: Dict[int, List[PatchInfo]] = {cid: [] for cid in fg_ids}
    candidates_any: List[PatchInfo] = []

    H, W = full_mask_level.shape[:2]

    for p in patches:
        x0, y0 = int(p.x0), int(p.y0)
        x1 = min(x0 + int(patch_size), W)
        y1 = min(y0 + int(patch_size), H)
        if x1 <= x0 or y1 <= y0:
            continue
        tile_mask = full_mask_level[y0:y1, x0:x1]
        if tile_mask.size == 0:
            continue

        present = []
        for cid in fg_ids:
            if int((tile_mask == int(cid)).sum()) >= int(min_present_pixels):
                present.append(int(cid))

        for cid in present:
            candidates[cid].append(p)
        if present:
            candidates_any.append(p)

    # Compute balanced targets and select tiles
    avail = {cid: len(candidates[cid]) for cid in fg_ids}
    targets = _compute_balanced_targets(total=int(qc_tile_samples), avail=avail, class_ids=fg_ids)

    selected: List[PatchInfo] = []
    selected_ids = set()
    selected_counts: Dict[int, int] = {cid: 0 for cid in fg_ids}

    for cid in fg_ids:
        take = int(targets.get(cid, 0))
        if take <= 0:
            continue
        for p in candidates[cid][:take]:
            if p.idx in selected_ids:
                continue
            selected.append(p)
            selected_ids.add(p.idx)
            selected_counts[cid] += 1

    # Fill remaining slots
    if len(selected) < int(qc_tile_samples):
        for p in candidates_any:
            if len(selected) >= int(qc_tile_samples):
                break
            if p.idx in selected_ids:
                continue
            selected.append(p)
            selected_ids.add(p.idx)

    # Write evidence files
    for p in tqdm(selected, total=len(selected), desc="Write QC tile_samples", dynamic_ncols=True):
        patch_pil = wsi_backend.read_patch(p.x0, p.y0, patch_size, level=level)
        patch_rgb = np.array(patch_pil)

        x0, y0 = int(p.x0), int(p.y0)
        x1 = min(x0 + int(patch_size), W)
        y1 = min(y0 + int(patch_size), H)
        tile_mask = full_mask_level[y0:y1, x0:x1].astype(np.uint8)
        if tile_mask.shape[0] != int(patch_size) or tile_mask.shape[1] != int(patch_size):
            # pad to patch_size to match image (white padding already in read_patch)
            canvas = np.zeros((int(patch_size), int(patch_size)), dtype=np.uint8)
            canvas[: tile_mask.shape[0], : tile_mask.shape[1]] = tile_mask
            tile_mask = canvas

        stem = f"tile_{p.idx:06d}_L{level}_X{int(p.x0)}_Y{int(p.y0)}"
        img_path = str(Path(images_dir) / f"{stem}.jpg")
        pred_path = str(Path(pred_dir) / f"{stem}_pred.png")
        merge_path = str(Path(merge_dir) / f"{stem}_merge.png")

        patch_pil.save(img_path, format="JPEG", quality=90)
        cv2.imwrite(pred_path, tile_mask)
        _save_merge_with_legend(
            img_rgb=patch_rgb,
            pred=tile_mask,
            out_path=merge_path,
            classes=classes,
            palette=palette,
            opacity=float(opacity),
        )

    return QCTileSampleResult(
        qc_dir=str(qc_dir),
        images_dir=str(images_dir),
        pred_dir=str(pred_dir),
        merge_dir=str(merge_dir),
        selected_counts=selected_counts,
        selected_tiles=[p.idx for p in selected],
    )
