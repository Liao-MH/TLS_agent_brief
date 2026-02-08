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
    """
    return {
        "default_balanced": Preset(
            name="default_balanced",
            params={
                "max_thumb": 2048,
                "tissue_thresh": 0.10,
                "patch_size": 2048,
                "overlap": 1024,
                "min_object_area": 40000,
                "simplify_eps": 2.0,
                "qc_tile_samples": 12,
                "save_all_patches": False,
                "mask_level": 0,  # 0=base-level; >0 only if backend supports it (OpenSlide)
                # postprocess
                "tls_gc_proximity_px": 32,  # dilation radius in pixels (mask_level coordinates)
            },
        ),
        "low_memory": Preset(
            name="low_memory",
            params={
                "max_thumb": 2048,
                "tissue_thresh": 0.12,
                "patch_size": 1024,
                "overlap": 256,
                "min_object_area": 25000,
                "simplify_eps": 2.0,
                "qc_tile_samples": 12,
                "save_all_patches": False,
                "mask_level": 1,
                "tls_gc_proximity_px": 24,
            },
        ),
        "fast_lowres": Preset(
            name="fast_lowres",
            params={
                "max_thumb": 1536,
                "tissue_thresh": 0.12,
                "patch_size": 1024,
                "overlap": 256,
                "min_object_area": 30000,
                "simplify_eps": 3.0,
                "qc_tile_samples": 12,
                "save_all_patches": False,
                "mask_level": 2,
                "tls_gc_proximity_px": 24,
            },
        ),
        "robust_stain_shift": Preset(
            name="robust_stain_shift",
            params={
                "max_thumb": 2048,
                "tissue_thresh": 0.08,   # more inclusive
                "patch_size": 2048,
                "overlap": 1024,
                "min_object_area": 40000,
                "simplify_eps": 2.0,
                "qc_tile_samples": 12,
                "save_all_patches": False,
                "mask_level": 0,
                "tls_gc_proximity_px": 40,
            },
        ),
        "debug_all_patches": Preset(
            name="debug_all_patches",
            params={
                "max_thumb": 2048,
                "tissue_thresh": 0.10,
                "patch_size": 2048,
                "overlap": 1024,
                "min_object_area": 40000,
                "simplify_eps": 2.0,
                "qc_tile_samples": 12,
                "save_all_patches": True,
                "mask_level": 0,
                "tls_gc_proximity_px": 32,
            },
        ),
    }
