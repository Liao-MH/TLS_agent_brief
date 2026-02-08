# projects/morph_fusion/transforms.py

from __future__ import annotations

import os.path as osp
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mmengine.fileio import get_local_path
from mmseg.registry import TRANSFORMS

from .morph_channels import resolve_morph_channel_indices


@TRANSFORMS.register_module()
class LoadMorphologyFromFile:
    """
    Load morphology feature map from .npy or .npz, and store it in results['morph'].

    Expected morph array shapes:
      - (H, W)            -> treated as (H, W, 1)
      - (H, W, C)
      - (C, H, W)         -> will be transposed to (H, W, C)

    Path resolution:
      - If morph_dir is provided:
            morph_path = join(morph_dir, stem(img_path) + morph_ext)
      - Else:
            morph_path = stem(img_path) + morph_suffix + morph_ext   in the same dir as img_path
    """

    def __init__(
        self,
        channel_names: Sequence[str],
        morph_dir: Optional[str] = None,
        morph_suffix: str = '_morph',
        morph_ext: str = '.npy',
        npz_key: Optional[str] = None,
        clip_range: Optional[Tuple[float, float]] = (0.0, 1.0),
        to_float32: bool = True,
    ) -> None:
        self.channel_names = list(channel_names)
        self.channel_indices = resolve_morph_channel_indices(self.channel_names)

        self.morph_dir = morph_dir
        self.morph_suffix = morph_suffix
        self.morph_ext = morph_ext
        self.npz_key = npz_key
        self.clip_range = clip_range
        self.to_float32 = to_float32

    def _infer_morph_path(self, img_path: str) -> str:
        stem = osp.splitext(osp.basename(img_path))[0]
        if self.morph_dir is not None:
            return osp.join(self.morph_dir, stem + self.morph_ext)
        # same dir as image
        img_dir = osp.dirname(img_path)
        return osp.join(img_dir, stem + self.morph_suffix + self.morph_ext)

    def _load_array(self, path: str) -> np.ndarray:
        with get_local_path(path) as local_path:
            if local_path.endswith('.npz'):
                npz = np.load(local_path)
                if self.npz_key is not None:
                    arr = npz[self.npz_key]
                else:
                    # pick the first key deterministically
                    keys = sorted(list(npz.keys()))
                    if not keys:
                        raise ValueError(f'Empty npz file: {path}')
                    arr = npz[keys[0]]
            else:
                arr = np.load(local_path, allow_pickle=False)
        return arr

    def _ensure_hwc(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            arr = arr[:, :, None]
        elif arr.ndim == 3:
            # allow CHW
            if arr.shape[0] < 16 and arr.shape[0] != arr.shape[1] and arr.shape[0] != arr.shape[2]:
                # heuristic: if first dim looks like channel
                # (C, H, W) -> (H, W, C)
                arr = np.transpose(arr, (1, 2, 0))
        else:
            raise ValueError(f'Unsupported morph array ndim={arr.ndim}, shape={arr.shape}')
        return arr

    def _select_channels(self, arr_hwc: np.ndarray) -> np.ndarray:
        c_total = arr_hwc.shape[2]
        max_idx = max(self.channel_indices)
        if max_idx >= c_total:
            raise ValueError(
                f'Morph array has {c_total} channels, but requested channel index {max_idx}. '
                f'channel_names={self.channel_names}, indices={self.channel_indices}'
            )
        return arr_hwc[:, :, self.channel_indices]

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img_path = results.get('img_path', None)
        if img_path is None:
            raise KeyError("results missing 'img_path' (required by LoadMorphologyFromFile).")

        morph_path = self._infer_morph_path(img_path)
        arr = self._load_array(morph_path)
        arr = self._ensure_hwc(arr)
        arr = self._select_channels(arr)

        if self.to_float32:
            arr = arr.astype(np.float32, copy=False)

        if self.clip_range is not None:
            lo, hi = self.clip_range
            arr = np.clip(arr, lo, hi)

        results['morph_path'] = morph_path
        results['morph'] = arr
        results['morph_channel_names'] = list(self.channel_names)
        return results

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'channel_names={self.channel_names}, morph_dir={self.morph_dir}, '
            f'morph_suffix={self.morph_suffix!r}, morph_ext={self.morph_ext!r}, '
            f'npz_key={self.npz_key!r}, clip_range={self.clip_range}, to_float32={self.to_float32})'
        )


@TRANSFORMS.register_module()
class ConcatMorphToImage:
    """Concat results['morph'] to results['img'] along channel dim (H, W, C)."""

    def __init__(self, pop_morph: bool = False) -> None:
        self.pop_morph = pop_morph

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img = results.get('img', None)
        morph = results.get('morph', None)
        if img is None:
            raise KeyError("results missing 'img' (required by ConcatMorphToImage).")
        if morph is None:
            # allow running without morphology
            return results

        if img.ndim != 3:
            raise ValueError(f'img must be HWC, got ndim={img.ndim}, shape={img.shape}')
        if morph.ndim != 3:
            raise ValueError(f'morph must be HWC, got ndim={morph.ndim}, shape={morph.shape}')
        if img.shape[:2] != morph.shape[:2]:
            raise ValueError(
                f'img/morph spatial mismatch: img={img.shape}, morph={morph.shape}. '
                f'Check your morph generation alignment.'
            )

        # concat
        results['img'] = np.concatenate([img, morph], axis=2)
        # update img_shape so downstream transforms are consistent
        results['img_shape'] = results['img'].shape[:2]

        if self.pop_morph:
            results.pop('morph', None)
        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(pop_morph={self.pop_morph})'


@TRANSFORMS.register_module()
class NormalizeRGBOnly:
    """
    Normalize only the first 3 channels (RGB/BGR) of results['img'].
    The remaining morph channels are kept unchanged (recommended already in [0,1]).

    This avoids mmseg's default Normalize applying mean/std to morph channels.
    """

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
        to_rgb: bool = True,
    ) -> None:
        if len(mean) != 3 or len(std) != 3:
            raise ValueError('NormalizeRGBOnly expects mean/std length == 3.')
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
        img = results.get('img', None)
        if img is None:
            raise KeyError("results missing 'img' (required by NormalizeRGBOnly).")
        if img.ndim != 3 or img.shape[2] < 3:
            raise ValueError(f'img must be HWC with >=3 channels, got shape={img.shape}')

        img = img.astype(np.float32, copy=False)

        rgb = img[:, :, :3]
        rest = img[:, :, 3:] if img.shape[2] > 3 else None

        if self.to_rgb:
            # swap BGR->RGB
            rgb = rgb[:, :, ::-1]

        rgb = (rgb - self.mean) / self.std

        if rest is None:
            results['img'] = rgb
        else:
            results['img'] = np.concatenate([rgb, rest], axis=2)

        return results

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}(mean={self.mean.tolist()}, std={self.std.tolist()}, '
            f'to_rgb={self.to_rgb})'
        )

