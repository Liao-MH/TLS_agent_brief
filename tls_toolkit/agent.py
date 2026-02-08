from __future__ import annotations

from pathlib import Path
import shutil
from typing import Dict, Any, Optional, Tuple

import numpy as np
import tifffile

from .presets import get_presets
from .constants import SCHEMA_VERSION
from .utils import ensure_dir, now_run_id, write_json, read_json, Timer, SimpleLogger, sha256_file
from .errors import ClassifiedError, ResourceError, WSIReadError, InferenceError

from .wsi import WSIBackend
from .tissue import run_tissue
from .patches import plan_patches
from .model import load_mmseg_model
from .infer import infer_tiles
from .stitch import stitch_center_crop

from .postprocess import postprocess_tls_gc_logic
from .vectorize import vectorize_mask_to_geojson
from .qc_tiles import save_qc_tile_samples_after_postprocess
from .analysis import analyze_and_qc
from .report import render_report_md


def classify_exception(e: Exception) -> ClassifiedError:
    msg = str(e)
    low = msg.lower()

    if isinstance(e, WSIReadError) or "openslide" in low or "wsi" in low:
        return ClassifiedError(
            category="io",
            message=msg,
            suggestion="Check file path/permissions; if format unsupported, convert to OME-TIFF; ensure openslide/vips are installed.",
            detail=repr(e),
        )
    if isinstance(e, ResourceError) or "out of memory" in low or "oom" in low or "no space" in low:
        return ClassifiedError(
            category="resource",
            message=msg,
            suggestion="Try low_memory preset; reduce patch_size/overlap; increase mask_level; disable saving patches; ensure enough disk space.",
            detail=repr(e),
        )
    if isinstance(e, InferenceError) or "mmseg" in low or "checkpoint" in low:
        return ClassifiedError(
            category="dependency",
            message=msg,
            suggestion="Verify mmseg installation, config/ckpt paths, and CUDA/torch compatibility.",
            detail=repr(e),
        )
    return ClassifiedError(
        category="unknown",
        message=msg,
        suggestion="Inspect logs/errors.log; run with debug_all_patches preset for more evidence.",
        detail=repr(e),
    )


def pick_preset_initial(wsi_backend: WSIBackend) -> str:
    info = wsi_backend.info()
    if info.width * info.height > 12_000_000_000:
        return "fast_lowres"
    return "default_balanced"


def adjust_preset_on_failure(preset_name: str, classified: ClassifiedError) -> str:
    if classified.category == "resource":
        return "low_memory"
    if classified.category == "io":
        return preset_name
    return "robust_stain_shift"


def adjust_preset_on_qc(preset_name: str, qc: Dict[str, Any]) -> Optional[str]:
    flags = set(qc.get("qc_flags", []))
    score = int(qc.get("qc_score", 0))

    if "no_patches_after_tissue_filter" in flags:
        return "robust_stain_shift"
    if "tissue_too_low" in flags and score < 60:
        return "robust_stain_shift"
    if "tls_fraction_high" in flags and score < 80:
        return "robust_stain_shift"
    if any(f.startswith("inference_failed_") for f in flags) and score < 70:
        return "low_memory"
    return None


def _get(params: Dict[str, Any], key: str, default: Any) -> Any:
    return params[key] if key in params else default


def run_pipeline_once(
    wsi_path: str,
    out_root: str,
    device: str,
    config_file: str,
    checkpoint: str,
    preset_name: str,
    preset_requested: Optional[str],
    allow_preset_fallback: bool,
    max_reruns_requested: int,
    max_reruns_effective: int,
) -> Tuple[str, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    presets = get_presets()
    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available={list(presets.keys())}")

    params = presets[preset_name].params

    slide_id = Path(wsi_path).stem.replace(" ", "_")
    run_id = now_run_id()
    run_dir = str(Path(out_root) / slide_id / "runs" / run_id)
    ensure_dir(run_dir)
    ensure_dir(Path(run_dir) / "logs")
    ensure_dir(Path(run_dir) / "meta")
    ensure_dir(Path(run_dir) / "artifacts")

    logger = SimpleLogger(str(Path(run_dir) / "logs" / "run.log"))
    logger.log(
        f"[INIT] slide_id={slide_id} run_id={run_id} preset={preset_name} "
        f"preset_requested={preset_requested} "
        f"fallback={'enabled' if allow_preset_fallback else 'disabled'} "
        f"max_reruns=requested:{max_reruns_requested} effective:{max_reruns_effective}"
    )

    run_meta: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "slide_id": slide_id,
        "run_id": run_id,
        "wsi_path": wsi_path,
        "out_root": out_root,
        "device": device,
        "preset": preset_name,
        "preset_params": params,
        "preset_requested": preset_requested,
        "preset_fallback_enabled": bool(allow_preset_fallback),
        "max_reruns_requested": int(max_reruns_requested),
        "max_reruns_effective": int(max_reruns_effective),
        "timing_s": {},
    }

    with Timer("wsi_open") as t:
        backend = WSIBackend(wsi_path)
        wsi_info = backend.info().__dict__
    run_meta["timing_s"]["wsi_open"] = t.elapsed_s
    run_meta["wsi"] = wsi_info
    logger.log(
        f"[WSI] backend={wsi_info['backend']} size={wsi_info['width']}x{wsi_info['height']} "
        f"levels={wsi_info['level_count']}"
    )

    mask_level = int(_get(params, "mask_level", 0))
    mask_level = max(0, min(mask_level, int(wsi_info["level_count"]) - 1))
    level_dim = backend.get_level_dim(mask_level)
    level_downsample = float(wsi_info["level_downsamples"][mask_level])

    # --------------------
    # Tissue mask (thumb)
    # --------------------
    with Timer("tissue") as t:
        max_thumb = int(_get(params, "max_thumb", 2048))

        # NEW params (with sensible defaults)
        tissue_min_region_frac = float(_get(params, "tissue_min_region_frac", 1.0 / 40.0))
        tissue_exclude_black_and_ink = bool(_get(params, "tissue_exclude_black_and_ink", True))
        tissue_otsu_delta = int(_get(params, "tissue_otsu_delta", 15))
        tissue_sat_thresh = int(_get(params, "tissue_sat_thresh", 20))
        tissue_close_ksize = int(_get(params, "tissue_close_ksize", 5))
        tissue_close_iter = int(_get(params, "tissue_close_iter", 2))
        tissue_open_ksize = int(_get(params, "tissue_open_ksize", 3))
        tissue_open_iter = int(_get(params, "tissue_open_iter", 1))

        # Call build_tissue_mask with backward compatibility
        try:
            tissue_mask, base_to_thumb_ds, thumb = backend.build_tissue_mask(
                max_thumb=max_thumb,
                min_region_frac=tissue_min_region_frac,
                exclude_black_and_ink=tissue_exclude_black_and_ink,
                otsu_delta=tissue_otsu_delta,
                sat_thresh=tissue_sat_thresh,
                close_ksize=tissue_close_ksize,
                close_iter=tissue_close_iter,
                open_ksize=tissue_open_ksize,
                open_iter=tissue_open_iter,
            )
        except TypeError:
            # old wsi.py signature fallback
            tissue_mask, base_to_thumb_ds, thumb = backend.build_tissue_mask(max_thumb=max_thumb)

        thumb_rgb = np.array(thumb)
        tissue_res = run_tissue(tissue_mask=tissue_mask, thumb_rgb=thumb_rgb, out_dir=run_dir)
        tissue_qc = tissue_res.qc
        tissue_qc["base_to_thumb_ds"] = float(base_to_thumb_ds)

        # for auditing
        run_meta["tissue_params_effective"] = {
            "max_thumb": max_thumb,
            "tissue_min_region_frac": tissue_min_region_frac,
            "tissue_exclude_black_and_ink": tissue_exclude_black_and_ink,
            "tissue_otsu_delta": tissue_otsu_delta,
            "tissue_sat_thresh": tissue_sat_thresh,
            "tissue_close_ksize": tissue_close_ksize,
            "tissue_close_iter": tissue_close_iter,
            "tissue_open_ksize": tissue_open_ksize,
            "tissue_open_iter": tissue_open_iter,
        }

    run_meta["timing_s"]["tissue"] = t.elapsed_s
    logger.log(f"[TISSUE] tissue_fraction_thumb={tissue_qc['tissue_fraction_thumb']:.4f}")

    # --------------------
    # Patch plan
    # --------------------
    with Timer("patch_plan") as t:
        patches = plan_patches(
            level_dim=level_dim,
            level=mask_level,
            patch_size=int(_get(params, "patch_size", 2048)),
            overlap=int(_get(params, "overlap", 1024)),
            tissue_mask_thumb=tissue_mask,
            base_to_thumb_ds=float(base_to_thumb_ds),
            level_downsample=float(level_downsample),
            tissue_thresh=float(_get(params, "tissue_thresh", 0.10)),
            out_dir=run_dir,
        )
    run_meta["timing_s"]["patch_plan"] = t.elapsed_s
    logger.log(f"[PATCH] num_patches={len(patches)} level={mask_level} level_dim={level_dim}")

    patch_plan = read_json(str(Path(run_dir) / "artifacts" / "analysis" / "patch_plan.json"))

    # --------------------
    # Load model
    # --------------------
    with Timer("load_model") as t:
        mb = load_mmseg_model(config_file=config_file, checkpoint=checkpoint, device=device)
    run_meta["timing_s"]["load_model"] = t.elapsed_s
    run_meta["model"] = mb.meta
    logger.log("[MODEL] loaded")

    if config_file:
        config_copy_path = Path(run_dir) / "meta" / "config.py"
        shutil.copyfile(config_file, config_copy_path)
        run_meta["config"] = {
            "path": config_file,
            "sha256": sha256_file(config_file),
            "copied_to": str(config_copy_path),
        }

    # --------------------
    # Inference
    # --------------------
    with Timer("inference") as t:
        pred_by_idx, infer_stats = infer_tiles(
            wsi_backend=backend,
            model=mb.model,
            patches=patches,
            patch_size=int(_get(params, "patch_size", 2048)),
            level=mask_level,
            out_dir=run_dir,
            save_all_patches=bool(_get(params, "save_all_patches", False)),
            qc_tile_samples=int(_get(params, "qc_tile_samples", 12)),
            enable_disk_fallback=True,
            config_file=config_file,
        )
    run_meta["timing_s"]["inference"] = t.elapsed_s
    infer_stats_dict = {
        "processed": infer_stats.processed,
        "failed": infer_stats.failed,
        "first_error": infer_stats.first_error,
        "first_error_traceback": getattr(infer_stats, "first_error_traceback", None),
        "class_pixel_counts": infer_stats.class_pixel_counts,
    }
    logger.log(f"[INFER] processed={infer_stats.processed} failed={infer_stats.failed}")

    if infer_stats.processed == 0:
        msg = infer_stats.first_error or "Inference failed for all patches."
        logger.log(f"[INFER] FATAL: processed=0. first_error={msg}")
        raise InferenceError(f"All patches failed during inference. first_error={msg}")

    # --------------------
    # Stitch
    # --------------------
    with Timer("stitch") as t:
        full_mask_path, stitch_meta = stitch_center_crop(
            patches=patches,
            pred_by_idx=pred_by_idx,
            level_dim=level_dim,
            out_dir=run_dir,
            patch_size=int(_get(params, "patch_size", 2048)),
            overlap=int(_get(params, "overlap", 1024)),
            mask_level=mask_level,
        )
    run_meta["timing_s"]["stitch"] = t.elapsed_s
    stitch_meta_dict = stitch_meta.__dict__
    logger.log(f"[STITCH] full_mask={full_mask_path}")

    # --------------------
    # Postprocess
    # --------------------
    with Timer("postprocess") as t:
        full_mask = tifffile.imread(full_mask_path)

        pp = postprocess_tls_gc_logic(
            full_mask=full_mask,
            tls_label=1,
            gc_label=2,
            immune_label=3,
            min_object_area=int(_get(params, "min_object_area", 40000)),
            touch_px=int(_get(params, "tls_gc_touch_px", 1)),
            expand_px=int(_get(params, "tls_expand_px", 5)),
            smooth_close_r=int(_get(params, "smooth_close_r", 2)),
            smooth_open_r=int(_get(params, "smooth_open_r", 1)),
        )

        tifffile.imwrite(full_mask_path, pp.mask.astype(np.uint8), dtype=np.uint8, bigtiff=True)

    run_meta["timing_s"]["postprocess"] = t.elapsed_s
    postproc_stats = pp.stats
    logger.log(f"[POST] tls_components_to_immune={postproc_stats.get('tls_components_to_immune', 0)}")

    # QC tile samples AFTER postprocess
    try:
        qc_tile_samples = int(_get(params, "qc_tile_samples", 12))
        if qc_tile_samples > 0:
            qc_res = save_qc_tile_samples_after_postprocess(
                wsi_backend=backend,
                model=mb.model,
                patches=patches,
                full_mask_level=pp.mask,
                patch_size=int(_get(params, "patch_size", 2048)),
                level=mask_level,
                out_dir=run_dir,
                qc_tile_samples=qc_tile_samples,
            )
            run_meta["qc_tile_samples"] = {
                "qc_dir": qc_res.qc_dir,
                "selected_counts": qc_res.selected_counts,
                "selected_tiles": qc_res.selected_tiles,
            }
            logger.log(f"[QC_SAMPLES] saved in {qc_res.qc_dir} counts={qc_res.selected_counts}")
    except Exception as e:
        logger.log(f"[QC_SAMPLES] WARN: failed to write tile_samples after postprocess: {e}")

    # --------------------
    # Vectorize
    # --------------------
    with Timer("vectorize") as t:
        vz = vectorize_mask_to_geojson(
            full_mask=pp.mask,
            out_dir=run_dir,
            min_object_area=int(_get(params, "min_object_area", 40000)),
            coord_scale=float(level_downsample),
        )
    run_meta["timing_s"]["vectorize"] = t.elapsed_s
    vector_stats = vz.stats
    vector_stats["geojson_path"] = vz.geojson_path
    vector_stats["qupath_geojson_path"] = vz.qupath_geojson_path
    vector_stats["objects_table_path"] = vz.objects_table_path
    vector_stats["mask_level"] = int(mask_level)
    vector_stats["level_downsample"] = float(level_downsample)
    logger.log(
        f"[VEC] geojson={vz.geojson_path} qupath_geojson={vz.qupath_geojson_path} "
        f"objects={vector_stats.get('objects_total', 0)}"
    )

    # --------------------
    # Analysis + QC + Report
    # --------------------
    with Timer("analysis") as t:
        ar = analyze_and_qc(
            out_dir=run_dir,
            tissue_qc=tissue_qc,
            patch_plan=patch_plan,
            infer_stats=infer_stats_dict,
            stitch_meta=stitch_meta_dict,
            postproc_stats=postproc_stats,
            vector_stats=vector_stats,
            full_mask_path=full_mask_path,
        )
    run_meta["timing_s"]["analysis"] = t.elapsed_s
    summary = ar.summary
    qc = ar.qc
    logger.log(f"[QC] score={qc.get('qc_score')} flags={qc.get('qc_flags')}")

    decision_trace = {
        "schema_version": SCHEMA_VERSION,
        "slide_id": slide_id,
        "run_id": run_id,
        "preset": preset_name,
        "steps": [],
    }

    with Timer("report") as t:
        report_path = render_report_md(
            out_dir=run_dir,
            run_meta=run_meta,
            decision_trace=decision_trace,
            summary=summary,
            qc=qc,
        )
    run_meta["timing_s"]["report"] = t.elapsed_s
    logger.log(f"[REPORT] {report_path}")

    write_json(str(Path(run_dir) / "meta" / "run_meta.json"), run_meta)
    write_json(str(Path(run_dir) / "meta" / "decision_trace.json"), decision_trace)

    return run_dir, run_meta, decision_trace, summary, qc


def run_agent(
    wsi_path: str,
    out_root: str,
    device: str,
    config_file: str,
    checkpoint: str,
    max_reruns: int = 1,
    preset: Optional[str] = None,
    allow_preset_fallback: bool = False,
) -> str:
    tmp_log = SimpleLogger(str(Path(out_root) / "agent_tmp.log"))
    tmp_log.log(f"[AGENT] start wsi={wsi_path}")

    backend = WSIBackend(wsi_path)
    presets = get_presets()

    preset_requested: Optional[str] = preset
    if preset_requested is not None and preset_requested not in presets:
        raise ValueError(f"Unknown preset: {preset_requested}. Available={list(presets.keys())}")

    if preset_requested:
        preset_name = preset_requested
        max_reruns_requested = int(max_reruns)
        if allow_preset_fallback and max_reruns_requested > 0:
            max_reruns_effective = max_reruns_requested
        else:
            max_reruns_effective = 0
    else:
        preset_name = pick_preset_initial(backend)
        max_reruns_requested = int(max_reruns)
        max_reruns_effective = max(0, max_reruns_requested)

    tmp_log.log(
        f"[AGENT] preset_selected={preset_name} preset_requested={preset_requested} "
        f"fallback={'enabled' if allow_preset_fallback else 'disabled'} "
        f"max_reruns=requested:{max_reruns_requested} effective:{max_reruns_effective}"
    )

    best_qc_score = -1
    best_run_dir = ""
    best_qc = None
    attempts = []

    for attempt in range(0, max_reruns_effective + 1):
        try:
            run_dir, _, _, _, qc = run_pipeline_once(
                wsi_path=wsi_path,
                out_root=out_root,
                device=device,
                config_file=config_file,
                checkpoint=checkpoint,
                preset_name=preset_name,
                preset_requested=preset_requested,
                allow_preset_fallback=bool(allow_preset_fallback),
                max_reruns_requested=max_reruns_requested,
                max_reruns_effective=max_reruns_effective,
            )
            attempts.append((run_dir, preset_name, qc))

            score = int(qc.get("qc_score", 0))
            if score > best_qc_score:
                best_qc_score = score
                best_run_dir = run_dir
                best_qc = qc

            if preset_requested and not (allow_preset_fallback and max_reruns_effective > 0):
                break
            next_preset = adjust_preset_on_qc(preset_name, qc)
            if next_preset is None or attempt == max_reruns_effective:
                break
            preset_name = next_preset

        except Exception as e:
            ce = classify_exception(e)
            if preset_requested and not (allow_preset_fallback and max_reruns_effective > 0):
                raise

            next_preset = adjust_preset_on_failure(preset_name, ce)
            if ce.category == "io":
                raise
            preset_name = next_preset

    if not best_run_dir:
        raise RuntimeError("Agent failed to produce any successful run.")

    trace_path = Path(best_run_dir) / "meta" / "decision_trace.json"
    meta_path = Path(best_run_dir) / "meta" / "run_meta.json"
    run_meta = read_json(str(meta_path))
    trace = read_json(str(trace_path))

    steps = []
    for i, (rd, pr, qc) in enumerate(attempts, 1):
        steps.append({
            "action": f"attempt_{i}_run",
            "rationale": "Deterministic preset execution; evaluate QC for possible rerun.",
            "evidence": {
                "run_dir": rd,
                "preset": pr,
                "qc_score": qc.get("qc_score"),
                "qc_flags": qc.get("qc_flags"),
            }
        })

    trace["steps"] = steps
    trace["selected_run_dir"] = best_run_dir
    trace["selected_qc_score"] = best_qc_score
    trace["agent_policy"] = {
        "preset_requested": preset_requested,
        "preset_fallback_enabled": bool(allow_preset_fallback),
        "max_reruns_requested": int(max_reruns_requested),
        "max_reruns_effective": int(max_reruns_effective),
    }

    flags = set(best_qc.get("qc_flags", [])) if best_qc else set()
    recs = []
    if "tissue_too_low" in flags or "tissue_low" in flags:
        recs.append("Tissue appears sparse; consider verifying slide region or tissue mask parameters.")
    if any(f.startswith("inference_failed_") for f in flags):
        recs.append("Inference failures detected; consider low_memory preset or checking GPU stability.")
    if "gc_fraction_near_zero" in flags:
        recs.append("GC fraction is near zero; validate model suitability for this cohort/stain domain.")
    trace["recommendations"] = recs

    write_json(str(trace_path), trace)

    summary = read_json(str(Path(best_run_dir) / "artifacts" / "analysis" / "summary.json"))
    qc = read_json(str(Path(best_run_dir) / "artifacts" / "analysis" / "qc_metrics.json"))
    render_report_md(
        out_dir=best_run_dir,
        run_meta=run_meta,
        decision_trace=trace,
        summary=summary,
        qc=qc,
    )

    return best_run_dir
