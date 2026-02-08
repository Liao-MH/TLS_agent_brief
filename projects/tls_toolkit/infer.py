# tls_toolkit/infer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import cv2
import torch
import traceback
from tqdm import tqdm
from PIL import Image

import matplotlib
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


def _update_counts(counts: Dict[int, int], pred: np.ndarray) -> None:
    vals, cts = np.unique(pred, return_counts=True)
    for v, c in zip(vals.tolist(), cts.tolist()):
        counts[int(v)] = counts.get(int(v), 0) + int(c)


def _prepare_test_pipeline_for_ndarray(model: Any) -> Any:
    """
    Build a stable mmseg Compose pipeline for ndarray inference that reuses model.cfg.test_pipeline,
    but forces the first loader to be LoadImageFromNDArray.

    This is the key to make train/val/test config reusable for WSI inference.
    """
    try:
        from mmengine.dataset import Compose  # type: ignore
    except Exception as e:
        raise InferenceError(f"mmengine is not available (Compose): {e}")

    cfg = getattr(model, "cfg", None)
    if cfg is None:
        raise InferenceError("Model has no cfg attached. init_model should attach model.cfg.")

    test_pipeline = None
    if hasattr(cfg, "test_pipeline"):
        test_pipeline = cfg.test_pipeline
    elif isinstance(cfg, dict):
        test_pipeline = cfg.get("test_pipeline", None)

    if test_pipeline is None:
        raise InferenceError("config has no test_pipeline; cannot reuse config for inference.")

    # Convert to list[dict]
    pipeline = [dict(x) for x in list(test_pipeline)]

    # Force first step to NDArray loader
    # - if it's LoadImageFromFile, replace
    # - if it's already NDArray loader, keep
    if len(pipeline) == 0:
        raise InferenceError("Empty test_pipeline in config.")

    first = pipeline[0]
    t = first.get("type", "")
    if t != "LoadImageFromNDArray":
        pipeline[0] = {"type": "LoadImageFromNDArray"}

    return Compose(pipeline)


@torch.no_grad()
def _infer_one_tile_with_pipeline(
    model: Any,
    pipeline: Any,
    patch_bgr: np.ndarray,
    img_path_hint: str,
) -> np.ndarray:
    """
    Run one tile inference with:
      - config test_pipeline (patched to NDArray loader)
      - model.test_step (therefore using model.data_preprocessor)
    """
    try:
        from mmengine.dataset import default_collate  # type: ignore
    except Exception:
        # fallback import path for some mmengine versions
        try:
            from mmengine.dataset.utils import default_collate  # type: ignore
        except Exception as e:
            raise InferenceError(f"Cannot import default_collate from mmengine: {e}")

    # results dict must be stable; some transforms access img_path
    results = {
        "img": patch_bgr,
        "img_path": img_path_hint,
    }

    sample = pipeline(results)
    batch = default_collate([sample])

    # mmseg models return list[SegDataSample]
    out = model.test_step(batch)
    if not out or not hasattr(out[0], "pred_sem_seg"):
        raise InferenceError("model.test_step returned unexpected output (no pred_sem_seg).")
    pred = out[0].pred_sem_seg.data
    # pred shape: (1,H,W) or (H,W)
    if pred.dim() == 3:
        pred = pred[0]
    return pred.detach().cpu().numpy().astype(np.uint8)


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
) -> Tuple[Dict[int, np.ndarray], InferStats]:
    """
    Patch-wise WSI inference that *reuses config*:
      - Uses model.cfg.test_pipeline (patched to NDArray loader)
      - Uses model.test_step => data_preprocessor runs as in training/test

    This makes config/pth directly reusable for inference and fixes first_error='img'.
    """
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

    classes, palette = _get_classes_palette(model)

    # Build pipeline ONCE
    pipeline = _prepare_test_pipeline_for_ndarray(model)

    for p in tqdm(patches, total=len(patches), desc="Infer tiles", dynamic_ncols=True):
        try:
            patch_pil = wsi_backend.read_patch(p.x0, p.y0, patch_size, level=level)
            patch_rgb = np.array(patch_pil)
            patch_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)

            pred = _infer_one_tile_with_pipeline(
                model=model,
                pipeline=pipeline,
                patch_bgr=patch_bgr,
                img_path_hint=f"tile_{p.idx:06d}_L{level}_X{p.x0}_Y{p.y0}.png",
            )

            pred_by_idx[p.idx] = pred
            _update_counts(class_counts, pred)

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
    return pred_by_idx, stats
