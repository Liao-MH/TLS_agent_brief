# projects/morph_inputfusion/transforms.py (新增/替换类)
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union, List
from pathlib import Path
import os
import numpy as np
import cv2

from mmseg.registry import TRANSFORMS

from .channel_spec import CHANNEL_SPECS
from .morph_features import (
    MorphComputeCfg,
    compute_morph_array,
    CHANNEL_ORDER_DEFAULT,
)

_IDX2NAME = {v: k for k, v in CHANNEL_SPECS.items()}


def _build_channel_mask(
    n_channels: int,
    channel_names: Optional[Sequence[str]],
    active_channel_names: Optional[Sequence[str]],
    channel_mask: Optional[Sequence[Union[int, float]]],
) -> np.ndarray:
    if channel_mask is not None:
        m = np.asarray(channel_mask, dtype=np.float32)
        if m.ndim != 1 or m.shape[0] != n_channels:
            raise ValueError(f'channel_mask length {m.shape} != n_channels={n_channels}')
        return (m > 0).astype(np.float32)

    if active_channel_names is None:
        return np.ones((n_channels,), dtype=np.float32)

    active = set(active_channel_names)

    if channel_names is not None:
        m = np.array([1.0 if name in active else 0.0 for name in channel_names], dtype=np.float32)
        if m.shape[0] != n_channels:
            raise ValueError(f'Internal error: mask len {m.shape[0]} != n_channels={n_channels}')
        return m

    m = np.zeros((n_channels,), dtype=np.float32)
    for i in range(n_channels):
        name = _IDX2NAME.get(i, None)
        if name is not None and name in active:
            m[i] = 1.0
    return m


@TRANSFORMS.register_module()
class LoadImageFromNDArrayTLS:
    """Load image from ndarray already stored in results['img']."""

    def __init__(self, to_float32: bool = False) -> None:
        self.to_float32 = bool(to_float32)

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return self.transform(results)

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if 'img' not in results:
            raise KeyError("LoadImageFromNDArrayTLS expects results['img'].")

        img = results['img']
        if img is None or img.ndim != 3:
            raise ValueError(f"Invalid results['img'] shape={None if img is None else img.shape}")

        if self.to_float32:
            img = img.astype(np.float32, copy=False)

        results['img'] = img
        results.setdefault('img_path', '')
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        results.setdefault('img_fields', ['img'])
        return results


def _resolve_morph_path(
    img_path: str,
    img_dirname: str,
    morph_dirname: str,
    morph_root: Optional[str],
) -> Path:
    img_path_obj = Path(img_path)
    if morph_root is not None:
        rel = img_path_obj.name if img_path_obj.is_absolute() else img_path_obj
        return (Path(morph_root) / rel).with_suffix(".npy")
    img_str = str(img_path_obj)
    marker = f"{os.sep}{img_dirname}{os.sep}"
    if marker in img_str:
        morph_str = img_str.replace(marker, f"{os.sep}{morph_dirname}{os.sep}")
        return Path(morph_str).with_suffix(".npy")
    return img_path_obj.with_suffix(".npy")


def _ensure_morph_hwc(morph: np.ndarray, expected_num_channels: Optional[int]) -> np.ndarray:
    if morph.ndim != 3:
        raise ValueError(f"Invalid morph shape={morph.shape}")
    if expected_num_channels is not None and morph.shape[2] == expected_num_channels:
        return morph
    if expected_num_channels is not None and morph.shape[0] == expected_num_channels:
        return np.transpose(morph, (1, 2, 0))
    if expected_num_channels is None and morph.shape[0] <= 8 and morph.shape[2] > 8:
        return np.transpose(morph, (1, 2, 0))
    return morph




@TRANSFORMS.register_module()
class PhotoMetricDistortionRGB:
    """Photometric distortion for *RGB channels only*.

    Designed for Morph input-fusion setting:
    - input results['img'] may be HWC with C==3 (pure RGB/BGR image), or C>=3 (RGB + morph channels).
    - This transform will only apply distortion on the first 3 channels.
    - Other channels (e.g., morph 5 channels) are kept unchanged.

    Notes:
    - The upstream pipeline typically uses LoadImageFromFile (uint8, BGR in mmcv),
      and cfg.data_preprocessor may later do bgr_to_rgb. Here we treat the first 3 channels
      as "color channels" without caring about semantic ordering; distortion is channel-wise
      and consistent.
    """

    def __init__(
        self,
        brightness_delta: int = 32,
        contrast_range: Tuple[float, float] = (0.5, 1.5),
        saturation_range: Tuple[float, float] = (0.5, 1.5),
        hue_delta: int = 18,
    ) -> None:
        self.brightness_delta = int(brightness_delta)
        self.contrast_lower = float(contrast_range[0])
        self.contrast_upper = float(contrast_range[1])
        self.saturation_lower = float(saturation_range[0])
        self.saturation_upper = float(saturation_range[1])
        self.hue_delta = int(hue_delta)

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return self.transform(results)

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if "img" not in results:
            raise KeyError("PhotoMetricDistortionRGB expects results['img'].")

        img = results["img"]
        if img is None or img.ndim != 3 or img.shape[2] < 3:
            raise ValueError(f"Invalid results['img'] shape={None if img is None else img.shape}")

        # Work on a float32 copy of the first 3 channels only
        img_f = img.astype(np.float32, copy=False)
        rgb = img_f[:, :, :3].copy()  # (H,W,3)
        extra = img_f[:, :, 3:] if img_f.shape[2] > 3 else None

        # random brightness
        if np.random.randint(2):
            delta = np.random.uniform(-self.brightness_delta, self.brightness_delta)
            rgb = rgb + delta

        # contrast may be applied either before or after HSV transform
        mode = np.random.randint(2)
        if mode == 1 and np.random.randint(2):
            alpha = np.random.uniform(self.contrast_lower, self.contrast_upper)
            rgb = rgb * alpha

        # saturation/hue in HSV space
        if np.random.randint(2):
            # rgb currently may be BGR depending on upstream; HSV conversion assumes BGR by default in OpenCV.
            # But distortion is still valid as long as consistent, so we use cv2.COLOR_BGR2HSV.
            # IMPORTANT: keep range [0,255] for cv2 conversion.
            rgb_clip = np.clip(rgb, 0, 255).astype(np.uint8)
            hsv = cv2.cvtColor(rgb_clip, cv2.COLOR_BGR2HSV).astype(np.float32)

            # saturation
            if np.random.randint(2):
                sat_scale = np.random.uniform(self.saturation_lower, self.saturation_upper)
                hsv[:, :, 1] = hsv[:, :, 1] * sat_scale

            # hue
            if np.random.randint(2):
                hue_delta = np.random.uniform(-self.hue_delta, self.hue_delta)
                hsv[:, :, 0] = hsv[:, :, 0] + hue_delta
                # wrap around [0, 179] for OpenCV hue
                hsv[:, :, 0] = np.mod(hsv[:, :, 0], 180.0)

            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32)

        # contrast after HSV if mode==0
        if mode == 0 and np.random.randint(2):
            alpha = np.random.uniform(self.contrast_lower, self.contrast_upper)
            rgb = rgb * alpha

        # clip back to valid image range (still float32; normalization later by preprocessor)
        rgb = np.clip(rgb, 0, 255).astype(np.float32)

        if extra is None:
            out = rgb
        else:
            # keep extra channels (morph) unchanged
            out = np.concatenate([rgb, extra.astype(np.float32, copy=False)], axis=2)

        results["img"] = out
        results["img_shape"] = out.shape[:2]
        return results


@TRANSFORMS.register_module()
class LoadMorphologyAndConcat:
    def __init__(
        self,
        enabled: bool = True,
        source: str = "file",
        channel_names: Optional[Sequence[str]] = None,
        active_channel_names: Optional[Sequence[str]] = None,
        channel_mask: Optional[Sequence[Union[int, float]]] = None,
        expected_num_channels: Optional[int] = 5,
        strict: bool = True,

        h_lo: float = 0.0,
        h_hi: float = 1.5,
        sobel_ksize: int = 3,
        lbp_enabled: bool = True,
        gabor_gamma: float = 0.5,
        gabor_sigma_1: float = 2.0,
        gabor_lambda_1: float = 6.0,
        gabor_sigma_2: float = 4.0,
        gabor_lambda_2: float = 12.0,
        clip: Optional[Tuple[float, float]] = (0.0, 1.0),
        img_dirname: str = "images",
        morph_dirname: str = "morph",
        morph_root: Optional[str] = None,

        **kwargs,
    ) -> None:
        self.enabled = bool(enabled)
        self.source = str(source)
        self.channel_names = list(channel_names) if channel_names is not None else list(CHANNEL_ORDER_DEFAULT)
        self.active_channel_names = list(active_channel_names) if active_channel_names is not None else None
        self.channel_mask = list(channel_mask) if channel_mask is not None else None
        self.expected_num_channels = int(expected_num_channels) if expected_num_channels is not None else None
        self.strict = bool(strict)
        self.img_dirname = str(img_dirname)
        self.morph_dirname = str(morph_dirname)
        self.morph_root = morph_root

        self.cfg = MorphComputeCfg(
            h_lo=float(h_lo),
            h_hi=float(h_hi),
            sobel_ksize=int(sobel_ksize),
            lbp_enabled=bool(lbp_enabled),
            gabor_gamma=float(gabor_gamma),
            gabor_sigma_1=float(gabor_sigma_1),
            gabor_lambda_1=float(gabor_lambda_1),
            gabor_sigma_2=float(gabor_sigma_2),
            gabor_lambda_2=float(gabor_lambda_2),
            clip_lo=float(clip[0]) if clip is not None else 0.0,
            clip_hi=float(clip[1]) if clip is not None else 1.0,
        )

    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return self.transform(results)

    def _enforce_expected_channels(self, morph: np.ndarray) -> np.ndarray:
        if self.expected_num_channels is None:
            return morph
        exp = int(self.expected_num_channels)
        c = int(morph.shape[2])
        if c == exp:
            return morph
        if c < exp:
            pad = np.zeros((morph.shape[0], morph.shape[1], exp - c), dtype=morph.dtype)
            return np.concatenate([morph, pad], axis=2)
        raise ValueError(f"morph has C={c} > expected_num_channels={exp}. Refuse to truncate.")

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return results

        if 'img' not in results:
            raise KeyError("LoadMorphologyAndConcat expects results['img'] from LoadImageFromFile/NDArray.")

        img = results['img']
        if img is None or img.ndim != 3 or img.shape[2] < 3:
            raise ValueError(f"Invalid results['img'] shape={None if img is None else img.shape}")

        if self.source == "compute":
            bgr = img[:, :, :3]
            morph = compute_morph_array(bgr, cfg=self.cfg, channel_order=self.channel_names, input_color="bgr")
        elif self.source == "file":
            img_path = results.get("img_path", None)
            if not img_path:
                raise KeyError("LoadMorphologyAndConcat(source='file') requires results['img_path'].")
            morph_path = _resolve_morph_path(
                img_path=str(img_path),
                img_dirname=self.img_dirname,
                morph_dirname=self.morph_dirname,
                morph_root=self.morph_root,
            )
            if not morph_path.exists():
                raise FileNotFoundError(f"Morphology .npy not found: {morph_path}")
            morph = np.load(str(morph_path))
            morph = _ensure_morph_hwc(morph, self.expected_num_channels)
            morph = morph.astype(np.float32, copy=False)
            morph = np.clip(morph, float(self.cfg.clip_lo), float(self.cfg.clip_hi))
        else:
            raise ValueError(f"Unknown morph source '{self.source}'. Expected 'file' or 'compute'.")

        morph = self._enforce_expected_channels(morph)

        c = int(morph.shape[2])
        m = _build_channel_mask(
            n_channels=c,
            channel_names=self.channel_names,
            active_channel_names=self.active_channel_names,
            channel_mask=self.channel_mask,
        ).astype(np.float32)
        morph = morph.astype(np.float32, copy=False) * m[None, None, :]

        img_f = img.astype(np.float32) if img.dtype != np.float32 else img
        out = np.concatenate([img_f[:, :, :3], morph], axis=2)

        results['img'] = out
        results['img_shape'] = out.shape[:2]
        results['morph_channel_names'] = self.channel_names
        results['morph_active_channel_names'] = self.active_channel_names
        results['morph_channel_mask'] = m.tolist()
        results['morph_expected_num_channels'] = self.expected_num_channels
        return results


@TRANSFORMS.register_module()
class ComputeMorphologyAndConcat(LoadMorphologyAndConcat):
    """Backward-compatible alias for LoadMorphologyAndConcat(source='compute')."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "source" not in kwargs:
            kwargs["source"] = "compute"
        super().__init__(*args, **kwargs)
