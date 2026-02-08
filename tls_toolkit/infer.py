from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import importlib

import numpy as np
import cv2
import torch
import traceback
from tqdm import tqdm
from PIL import Image
import matplotlib
# Robust for headless HPC / SLURM (no DISPLAY)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .patches import PatchInfo
from .utils import ensure_dir
from .errors import ResourceError, InferenceError


@dataclass
class InferStats:
    processed: int
    failed: int
    first_error: Optional[str]
    first_error_traceback: Optional[str]
    class_pixel_counts: Dict[int, int]

def _get_classes_palette(model: Any) -> Tuple[List[str], List[Tuple[int, int, int]]]:
    """
    Try to obtain classes/palette from model.dataset_meta (mmseg convention).
    Fallback to TLS toolkit defaults.
    """
    # mmseg often attaches dataset_meta with 'classes' and 'palette'
    meta = getattr(model, "dataset_meta", None)
    if isinstance(meta, dict):
        classes = meta.get("classes")
        palette = meta.get("palette")
        if isinstance(classes, (list, tuple)) and isinstance(palette, (list, tuple)) and len(classes) == len(palette):
            # palette may be list[list[int]] or list[tuple[int,int,int]]
            pal = []
            for c in palette:
                if isinstance(c, (list, tuple)) and len(c) == 3:
                    pal.append((int(c[0]), int(c[1]), int(c[2])))
            if len(pal) == len(classes):
                return list(map(str, classes)), pal

    # Fallback: 4 classes (background + 3 foreground)
    classes = ["background", "TLS", "GC", "ImmuneCluster"]
    palette = [
        (0, 0, 0),         # background
        (0, 255, 0),       # TLS
        (235, 143, 124),   # GC
        (0, 0, 255),       # ImmuneCluster
    ]
    return classes, palette


def _colorize_seg(pred: np.ndarray, palette: List[Tuple[int, int, int]]) -> np.ndarray:
    """
    Convert label map (H,W) to RGB image (H,W,3) using palette.
    Labels outside palette range are clipped to 0.
    """
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
    """
    Save an overlay visualization with legend to out_path.
    """
    seg_rgb = _colorize_seg(pred, palette)
    # match user's formula: seg*(1-opacity) + img*opacity
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



def _update_counts(counts: Dict[int, int], pred: np.ndarray) -> None:
    vals, cts = np.unique(pred, return_counts=True)
    for v, c in zip(vals.tolist(), cts.tolist()):
        counts[int(v)] = counts.get(int(v), 0) + int(c)

def _is_none_split_error(e: Exception) -> bool:
    """
    Heuristic for the mmseg/mmengine bug/assumption where filename is None and code does None.split(...).
    """
    msg = str(e)
    return ("NoneType" in msg) and ("split" in msg)


def _dominant_fg_class(pred: np.ndarray, fg_ids: Tuple[int, ...] = (1, 2, 3)) -> int:
    """
    Return the foreground class id with the most pixels among fg_ids.
    If no foreground pixels exist, return 0.
    """
    if pred.size == 0:
        return 0
    counts = {}
    for cid in fg_ids:
        counts[cid] = int((pred == cid).sum())
    best = max(counts.items(), key=lambda kv: kv[1])
    return int(best[0]) if best[1] > 0 else 0


def _compute_balanced_targets(
    total: int,
    avail: Dict[int, int],
    class_ids: Tuple[int, ...] = (1, 2, 3),
) -> Dict[int, int]:
    """
    Try to allocate total samples roughly equally across class_ids.
    If a class has insufficient availability, re-distribute remainder to others.
    """
    if total <= 0:
        return {cid: 0 for cid in class_ids}
    base = total // len(class_ids)
    rem = total % len(class_ids)
    targets = {cid: base for cid in class_ids}

    # distribute remainder to classes with more availability
    order = sorted(class_ids, key=lambda c: avail.get(c, 0), reverse=True)
    for i in range(rem):
        targets[order[i % len(order)]] += 1

    # clamp to availability and redistribute deficits
    deficit = 0
    for cid in class_ids:
        a = avail.get(cid, 0)
        if targets[cid] > a:
            deficit += targets[cid] - a
            targets[cid] = a

    if deficit > 0:
        # redistribute deficit to classes with remaining capacity
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


def _strip_annotations(pipeline_cfg: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove annotation-loading transforms for pure inference."""
    out = []
    for t in pipeline_cfg:
        if not isinstance(t, dict):
            out.append(t)
            continue
        if t.get("type") in ("LoadAnnotations", "LoadSegAnnotations"):
            continue
        out.append(t)
    return out


def _build_test_pipeline(config_file: Optional[str], model: Any, use_ndarray: bool) -> Any:
    try:
        from mmengine.config import Config  # type: ignore
        from mmengine.dataset import Compose, pseudo_collate  # type: ignore
    except Exception as e:
        raise InferenceError(f"mmengine is not available: {e}")

    if config_file is not None:
        cfg = Config.fromfile(config_file)
    else:
        cfg = getattr(model, "cfg", None)
        if cfg is None:
            raise InferenceError("Model does not expose cfg; please pass config_file.")

    # Ensure custom modules are imported (register TRANSFORMS/MODELS)
    custom_imports = cfg.get("custom_imports", None)
    if isinstance(custom_imports, dict):
        allow_failed = bool(custom_imports.get("allow_failed_imports", False))
        for mod in custom_imports.get("imports", []):
            try:
                importlib.import_module(mod)
            except Exception:
                if not allow_failed:
                    raise

    # Prefer infer_pipeline if present; fallback to test_pipeline
    if hasattr(cfg, "infer_pipeline") and cfg.get("infer_pipeline", None) is not None:
        pipe_cfg = list(cfg.infer_pipeline)
    else:
        pipe_cfg = list(cfg.test_pipeline)

    # Hard safety: inference must not require seg_map_path
    pipe_cfg = _strip_annotations(pipe_cfg)

    # NDArray input: replace first loader
    if use_ndarray and len(pipe_cfg) > 0 and isinstance(pipe_cfg[0], dict):
        first = dict(pipe_cfg[0])
        if first.get("type") == "LoadImageFromFile":
            first["type"] = "LoadImageFromNDArrayTLS"
        pipe_cfg[0] = first

    return Compose(pipe_cfg), pseudo_collate



def _run_model_test_step(model: Any, data: Dict[str, Any], pseudo_collate: Any) -> Any:
    data = pseudo_collate([data])
    with torch.no_grad():
        results = model.test_step(data)
    if isinstance(results, (list, tuple)) and len(results) > 0:
        return results[0]
    return results


def infer_tiles(
    wsi_backend: Any,
    model: Any,
    patches: List[PatchInfo],
    patch_size: int,
    level: int,
    out_dir: str,
    save_all_patches: bool = False,
    qc_tile_samples: int = 12,
    enable_disk_fallback: bool = True,
    config_file: Optional[str] = None,
) -> Tuple[Dict[int, np.ndarray], InferStats]:
    """
    Run inference per patch, return predictions in-memory keyed by patch idx.

    Default is I/O-light: saves only a limited number of tile samples for QC evidence.

    """
    pipeline, pseudo_collate = _build_test_pipeline(config_file=config_file, model=model, use_ndarray=True)

    out_dir = ensure_dir(out_dir)
    qc_dir = ensure_dir(Path(out_dir) / "artifacts" / "qc" / "tile_samples")
    patches_img_dir = ensure_dir(Path(qc_dir) / "images")
    patches_pred_dir = ensure_dir(Path(qc_dir) / "pred")
    patches_merge_dir = ensure_dir(Path(qc_dir) / "merge")
    pred_by_idx: Dict[int, np.ndarray] = {}
    class_counts: Dict[int, int] = {}

    processed = 0
    failed = 0
    first_error = None
    first_error_traceback = None

    # collect candidates for balanced sampling (store only up to qc_tile_samples per class to limit memory)
    classes, palette = _get_classes_palette(model)
    fg_ids = (1, 2, 3)
    max_pool_per_class = max(1, int(qc_tile_samples))
    candidates: Dict[int, List[Tuple[PatchInfo, np.ndarray, Image.Image]]] = {1: [], 2: [], 3: []}
    candidates_any: List[Tuple[PatchInfo, np.ndarray, Image.Image]] = []


    for p in tqdm(patches, total=len(patches), desc="Infer tiles", dynamic_ncols=True):
        try:
            patch_pil = wsi_backend.read_patch(p.x0, p.y0, patch_size, level=level)
            patch_rgb = np.array(patch_pil)
            patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)

            data = dict(img=patch_bgr)
            data = pipeline(data)
            result = _run_model_test_step(model=model, data=data, pseudo_collate=pseudo_collate)
            pred = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
            pred_by_idx[p.idx] = pred
            _update_counts(class_counts, pred)

            # Save-all remains unchanged
            if save_all_patches:
                img_path = str(Path(patches_img_dir) / f"tile_{p.idx:06d}_L{level}_X{p.x0}_Y{p.y0}.jpg")
                pred_path = str(Path(patches_pred_dir) / f"tile_{p.idx:06d}_pred.png")
                patch_pil.save(img_path, format="JPEG", quality=90)
                cv2.imwrite(pred_path, pred)
                merge_path = str(Path(patches_merge_dir) / f"tile_{p.idx:06d}_merge.png")
                _save_merge_with_legend(
                    img_rgb=patch_rgb,
                    pred=pred,
                    out_path=merge_path,
                    classes=classes,
                    palette=palette,
                    opacity=0.5,
                )

            # balanced tile_samples pool (dominant fg class)
            dom = _dominant_fg_class(pred, fg_ids=fg_ids)
            if dom in candidates and len(candidates[dom]) < max_pool_per_class:
                candidates[dom].append((p, pred, patch_pil.copy()))
            if len(candidates_any) < (max_pool_per_class * 3):
                candidates_any.append((p, pred, patch_pil.copy()))

            processed += 1

        except RuntimeError as e:
            msg = str(e)
            failed += 1
            if first_error is None:
                first_error = msg
                first_error_traceback = traceback.format_exc()
            if "out of memory" in msg.lower():
                raise ResourceError(f"CUDA OOM during inference: {msg}") from e
        except Exception as e:
            failed += 1
            if first_error is None:
                first_error = str(e)
                first_error_traceback = traceback.format_exc()

    stats = InferStats(
        processed=processed,
        failed=failed,
        first_error=first_error,
        first_error_traceback=first_error_traceback,
        class_pixel_counts=class_counts,
    )
    
    # NOTE:
    # QC tile_samples are now generated AFTER postprocess (agent.py),
    # so ImmuneCluster (rule-derived) can be correctly sampled.
    # Keep infer.py I/O-light by default.
    if False and (not save_all_patches) and int(qc_tile_samples) > 0:
        avail = {cid: len(candidates[cid]) for cid in fg_ids}
        targets = _compute_balanced_targets(total=int(qc_tile_samples), avail=avail, class_ids=fg_ids)

        selected: List[Tuple[PatchInfo, np.ndarray, Image.Image]] = []
        selected_ids = set()

        # primary: per-class allocation
        for cid in fg_ids:
            take = targets.get(cid, 0)
            if take <= 0:
                continue
            pool = candidates[cid][:take]
            for item in pool:
                selected.append(item)
                selected_ids.add(item[0].idx)

        # fill: if still short, take from candidates_any
        if len(selected) < int(qc_tile_samples):
            for item in candidates_any:
                if len(selected) >= int(qc_tile_samples):
                    break
                if item[0].idx in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(item[0].idx)

        # write selected qc samples (images + pred + merge)
        for p, pred, patch_pil in selected:
            patch_rgb = np.array(patch_pil)
            img_path = str(Path(patches_img_dir) / f"tile_{p.idx:06d}_L{level}_X{p.x0}_Y{p.y0}.jpg")
            pred_path = str(Path(patches_pred_dir) / f"tile_{p.idx:06d}_pred.png")
            merge_path = str(Path(patches_merge_dir) / f"tile_{p.idx:06d}_merge.png")
            patch_pil.save(img_path, format="JPEG", quality=90)
            cv2.imwrite(pred_path, pred)
            _save_merge_with_legend(
                img_rgb=patch_rgb,
                pred=pred,
                out_path=merge_path,
                classes=classes,
                palette=palette,
                opacity=0.5,
            )

    
    return pred_by_idx, stats
