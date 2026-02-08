from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import tifffile

from .constants import SCHEMA_VERSION
from .utils import ensure_dir, write_json


@dataclass
class AnalysisResult:
    summary_path: str
    qc_path: str
    summary: Dict[str, Any]
    qc: Dict[str, Any]


def _pixel_fractions(mask: np.ndarray) -> Dict[str, float]:
    total = int(mask.size)
    vals, cts = np.unique(mask, return_counts=True)
    out: Dict[str, float] = {}
    for v, c in zip(vals.tolist(), cts.tolist()):
        out[str(int(v))] = float(int(c) / total) if total > 0 else 0.0
    return out


def _qc_score_and_flags(
    tissue_fraction_thumb: float,
    num_patches: int,
    inference_failed: int,
    pixel_fracs: Dict[str, float],
    objects_total: int,
) -> Tuple[int, list]:
    flags = []
    score = 100

    if tissue_fraction_thumb < 0.02:
        flags.append("tissue_too_low")
        score -= 35
    elif tissue_fraction_thumb < 0.05:
        flags.append("tissue_low")
        score -= 15

    if num_patches == 0:
        flags.append("no_patches_after_tissue_filter")
        score -= 80

    if inference_failed > 0:
        flags.append(f"inference_failed_{inference_failed}")
        score -= min(40, 10 * inference_failed)

    tls_frac = pixel_fracs.get("1", 0.0)
    gc_frac = pixel_fracs.get("2", 0.0)

    if tls_frac > 0.25:
        flags.append("tls_fraction_high")
        score -= 10
    if tls_frac < 0.0001 and objects_total > 0:
        flags.append("tls_fraction_near_zero")
        score -= 10
    if gc_frac < 0.00005:
        flags.append("gc_fraction_near_zero")
        score -= 8

    score = max(0, min(100, score))
    return score, flags


def analyze_and_qc(
    out_dir: str,
    tissue_qc: Dict[str, Any],
    patch_plan: Dict[str, Any],
    infer_stats: Dict[str, Any],
    stitch_meta: Dict[str, Any],
    postproc_stats: Dict[str, Any],
    vector_stats: Dict[str, Any],
    full_mask_path: str,
) -> AnalysisResult:
    out_dir = ensure_dir(out_dir)
    analysis_dir = ensure_dir(Path(out_dir) / "artifacts" / "analysis")

    summary_path = str(Path(analysis_dir) / "summary.json")
    qc_path = str(Path(analysis_dir) / "qc_metrics.json")

    mask = tifffile.imread(full_mask_path)
    pixel_fracs = _pixel_fractions(mask)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "mask": {
            "path": full_mask_path,
            "mask_level": stitch_meta.get("mask_level"),
            "width": stitch_meta.get("width"),
            "height": stitch_meta.get("height"),
            "pixel_fractions": pixel_fracs,
        },
        "tissue": tissue_qc,
        "patch_plan": {
            "level": patch_plan.get("level"),
            "num_patches": patch_plan.get("num_patches"),
            "patch_size": patch_plan.get("patch_size"),
            "overlap": patch_plan.get("overlap"),
            "stride": patch_plan.get("stride"),
            "tissue_thresh": patch_plan.get("tissue_thresh"),
        },
        "inference": infer_stats,
        "stitch": stitch_meta,
        "postprocess": postproc_stats,
        "vectorize": vector_stats,
    }

    tissue_fraction_thumb = float(tissue_qc.get("tissue_fraction_thumb", 0.0))
    num_patches = int(patch_plan.get("num_patches", 0))
    inference_failed = int(infer_stats.get("failed", 0))
    objects_total = int(vector_stats.get("objects_total", 0))

    score, flags = _qc_score_and_flags(
        tissue_fraction_thumb=tissue_fraction_thumb,
        num_patches=num_patches,
        inference_failed=inference_failed,
        pixel_fracs=pixel_fracs,
        objects_total=objects_total,
    )

    qc = {
        "schema_version": SCHEMA_VERSION,
        "qc_score": int(score),
        "qc_flags": flags,
        "notes": "QC score is heuristic; refine thresholds per cohort and scanner domain.",
    }

    write_json(summary_path, summary)
    write_json(qc_path, qc)

    return AnalysisResult(
        summary_path=summary_path,
        qc_path=qc_path,
        summary=summary,
        qc=qc,
    )
