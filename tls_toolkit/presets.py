from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Preset:
    name: str
    params: Dict[str, Any]


def get_presets() -> Dict[str, Preset]:
    """
    Presets are intentionally explicit and auditable.
    Agent may choose among them and apply small deterministic adjustments.

    -------------------------
    PARAMETER REFERENCE
    -------------------------

    [WSI / Tissue mask]
      - max_thumb (int):
          Max thumbnail size (pixels) used to build tissue mask. Larger = more accurate, slower.
      - tissue_thresh (float):
          Tissue probability / fraction threshold for keeping a patch (thumb-space). Higher = stricter.

      (NEW) tissue-mask constraints you requested:
      - tissue_min_region_frac (float):
          Keep only connected components with area >= (thumb_area * tissue_min_region_frac).
          Example: 1/40 keeps only regions >= 2.5% of whole-slide thumbnail area.

      - tissue_exclude_black_and_ink (bool):
          If True, suppress meaningless black background and ink regions from tissue mask.

      - tissue_otsu_delta (int):
          Threshold adjustment: T_adj = T_otsu - tissue_otsu_delta.
          Larger delta => more inclusive for dark tissue (but risks including background).

      - tissue_sat_thresh (int):
          Saturation threshold used in tissue detection. Higher => stricter.

      - tissue_close_ksize (int), tissue_close_iter (int):
          Morph CLOSE kernel size / iterations for tissue mask.

      - tissue_open_ksize (int), tissue_open_iter (int):
          Morph OPEN kernel size / iterations for tissue mask.

    [Patch planning / Inference]
      - patch_size (int):
          Patch size in base-level pixel units of the selected mask_level (NOT always level0).
      - overlap (int):
          Patch overlap in same coordinate system as patch_size.
      - mask_level (int):
          Which WSI pyramid level to stitch/output mask at.
          0 = level0 (base); 1/2... = downsampled. Lower memory at higher levels.

      - save_all_patches (bool):
          Save every patch image & prediction (debug heavy).

      - qc_tile_samples (int):
          Number of QC tiles to sample AFTER postprocess. 0 disables.

    [Object filtering]
      - min_object_area (int):
          Minimum object area in mask_level pixel units.
          Used in:
            1) postprocess: remove small CCs / sanitize masks
            2) vectorize: ignore tiny contours
          Increase = fewer small annotations; decrease = keep more.

    [Postprocess: TLS/GC/Immune logic]  (NEW RULE)
      - tls_gc_touch_px (int):
          Touch tolerance radius (mask_level pixels). TLS CC is considered "touching GC"
          if dilate(TLS_CC, tls_gc_touch_px) overlaps GC.
          Larger makes it easier to be counted as TLS (fewer become ImmuneCluster).

      - tls_expand_px (int):
          TLS boundary expansion radius (mask_level pixels).
          TLS_final = dilate( TLS_CC ∪ touched_GC, tls_expand_px )
          Your requested default is 5.

      - smooth_close_r (int):
          Morphological close radius (mask_level pixels) applied in postprocess sanitize.
          Higher = smoother + more filled gaps, but can merge nearby structures.

      - smooth_open_r (int):
          Morphological open radius (mask_level pixels) applied in postprocess sanitize.
          Higher = removes spurs/noise, but can erode thin structures.

      NOTE: postprocess currently also performs "fill external contours" (hole removal)
      at the mask level to stabilize geometry. If you later want to toggle it, add a
      boolean param like `fill_external_contours`.

    [Vectorize / GeoJSON export]
      - coord_scale handled by agent (level_downsample). Not set here.

      - tls_includes_gc (bool):
          If True: TLS polygon geometry = (TLS mask OR GC mask)
          This enforces the semantic "every TLS polygon contains GC".
          If False: TLS polygon only uses TLS label pixels.

      - export_gc (bool):
          If True: export GC as its own class in geojson.
          If False: do not export GC objects (still may be included inside TLS geometry
          when tls_includes_gc=True).

      IMPORTANT: In the new vectorize.py I provided, we do NOT simplify polygons,
      to keep polygons consistent with mask. So `simplify_eps` is kept for backward
      compatibility but NOT used unless you reintroduce simplification.

    [Deprecated / legacy]
      - simplify_eps (float):
          Previously used for contour simplification. In the new strict "mask-consistent"
          export, simplification is disabled to guarantee polygon == mask boundary.
          Keep only for backward compatibility.

      - tls_gc_proximity_px (int):
          Old rule param (dilate TLS CC and check GC intersection).
          Deprecated if you use the new postprocess rule; kept to avoid breaking older logs.
    """
    return {
        "default_balanced": Preset(
            name="default_balanced",
            params={
                # -------------------------
                # WSI / Tissue
                # -------------------------
                "max_thumb": 2048,          # thumb max size (px)
                "tissue_thresh": 0.10,      # tissue filter threshold

                # (NEW) tissue mask constraints
                "tissue_min_region_frac": 1.0 / 40.0,       # keep CC area >= thumb_area/40
                "tissue_exclude_black_and_ink": True,       # remove black + ink
                "tissue_otsu_delta": 15,                    # T_adj = T_otsu - delta
                "tissue_sat_thresh": 20,                    # saturation threshold
                "tissue_close_ksize": 5,
                "tissue_close_iter": 2,
                "tissue_open_ksize": 3,
                "tissue_open_iter": 1,

                # -------------------------
                # Patch / Inference
                # -------------------------
                "patch_size": 2048,         # patch size @ mask_level coords
                "overlap": 1024,            # overlap @ mask_level coords
                "mask_level": 0,            # 0=level0; higher=downsampled
                "save_all_patches": False,  # debug switch
                "qc_tile_samples": 12,      # QC tiles after postprocess

                # -------------------------
                # Object filtering
                # -------------------------
                "min_object_area": 40000,   # min CC/contour area @ mask_level coords

                # -------------------------
                # Postprocess (NEW)
                # -------------------------
                "tls_gc_touch_px": 1,       # touch tolerance radius (px @ mask_level)
                "tls_expand_px": 5,         # TLS expand radius (px @ mask_level)
                "smooth_close_r": 2,        # close radius (px @ mask_level)
                "smooth_open_r": 4,         # open radius (px @ mask_level)

                # -------------------------
                # Vectorize / GeoJSON
                # -------------------------
                "tls_includes_gc": True,    # TLS geometry includes GC pixels
                "export_gc": True,          # export GC as its own objects

                # -------------------------
                # Deprecated / legacy
                # -------------------------
                "simplify_eps": 2.0,        # (deprecated) contour simplification
                "tls_gc_proximity_px": 32,  # (deprecated) old rule
            },
        ),

        "low_memory": Preset(
            name="low_memory",
            params={
                "max_thumb": 2048,
                "tissue_thresh": 0.12,

                # (NEW) tissue mask constraints
                "tissue_min_region_frac": 1.0 / 40.0,
                "tissue_exclude_black_and_ink": True,
                "tissue_otsu_delta": 15,
                "tissue_sat_thresh": 20,
                "tissue_close_ksize": 5,
                "tissue_close_iter": 2,
                "tissue_open_ksize": 3,
                "tissue_open_iter": 1,

                "patch_size": 1024,
                "overlap": 256,
                "mask_level": 1,
                "save_all_patches": False,
                "qc_tile_samples": 12,

                "min_object_area": 25000,

                # Postprocess (NEW): slightly more permissive touch under low-res
                "tls_gc_touch_px": 1,
                "tls_expand_px": 50,
                "smooth_close_r": 2,
                "smooth_open_r": 4,

                "tls_includes_gc": True,
                "export_gc": True,

                # Deprecated / legacy
                "simplify_eps": 2.0,
                "tls_gc_proximity_px": 24,
            },
        ),

        "fast_lowres": Preset(
            name="fast_lowres",
            params={
                "max_thumb": 1536,
                "tissue_thresh": 0.12,

                # (NEW) tissue mask constraints
                "tissue_min_region_frac": 1.0 / 40.0,
                "tissue_exclude_black_and_ink": True,
                "tissue_otsu_delta": 15,
                "tissue_sat_thresh": 20,
                "tissue_close_ksize": 5,
                "tissue_close_iter": 2,
                "tissue_open_ksize": 3,
                "tissue_open_iter": 1,

                "patch_size": 1024,
                "overlap": 256,
                "mask_level": 2,
                "save_all_patches": False,
                "qc_tile_samples": 12,

                "min_object_area": 30000,

                # Postprocess (NEW): often need a bit larger touch at low-res (optional)
                "tls_gc_touch_px": 1,
                "tls_expand_px": 5,
                "smooth_close_r": 2,
                "smooth_open_r": 1,

                "tls_includes_gc": True,
                "export_gc": True,

                # Deprecated / legacy
                "simplify_eps": 3.0,
                "tls_gc_proximity_px": 24,
            },
        ),

        "robust_stain_shift": Preset(
            name="robust_stain_shift",
            params={
                "max_thumb": 2048,
                "tissue_thresh": 0.08,   # more inclusive

                # (NEW) tissue mask constraints (more permissive, keep slightly smaller CCs)
                "tissue_min_region_frac": 1.0 / 60.0,
                "tissue_exclude_black_and_ink": True,
                "tissue_otsu_delta": 20,
                "tissue_sat_thresh": 15,
                "tissue_close_ksize": 7,
                "tissue_close_iter": 2,
                "tissue_open_ksize": 3,
                "tissue_open_iter": 1,

                "patch_size": 2048,
                "overlap": 1024,
                "mask_level": 0,
                "save_all_patches": False,
                "qc_tile_samples": 12,

                "min_object_area": 40000,

                # Postprocess (NEW): a bit more tolerant in challenging stains
                "tls_gc_touch_px": 2,    # more tolerant adjacency
                "tls_expand_px": 5,
                "smooth_close_r": 3,     # slightly stronger smoothing
                "smooth_open_r": 1,

                "tls_includes_gc": True,
                "export_gc": True,

                # Deprecated / legacy
                "simplify_eps": 2.0,
                "tls_gc_proximity_px": 40,
            },
        ),

        "debug_all_patches": Preset(
            name="debug_all_patches",
            params={
                "max_thumb": 2048,
                "tissue_thresh": 0.10,

                # (NEW) tissue mask constraints
                "tissue_min_region_frac": 1.0 / 40.0,
                "tissue_exclude_black_and_ink": True,
                "tissue_otsu_delta": 15,
                "tissue_sat_thresh": 20,
                "tissue_close_ksize": 5,
                "tissue_close_iter": 2,
                "tissue_open_ksize": 3,
                "tissue_open_iter": 1,

                "patch_size": 2048,
                "overlap": 1024,
                "mask_level": 0,
                "save_all_patches": True,   # heavy debug
                "qc_tile_samples": 12,

                "min_object_area": 40000,

                "tls_gc_touch_px": 1,
                "tls_expand_px": 5,
                "smooth_close_r": 2,
                "smooth_open_r": 1,

                "tls_includes_gc": True,
                "export_gc": True,

                # Deprecated / legacy
                "simplify_eps": 2.0,
                "tls_gc_proximity_px": 32,
            },
        ),
    }
