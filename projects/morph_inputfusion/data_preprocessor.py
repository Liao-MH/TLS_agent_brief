# projects/morph_inputfusion/data_preprocessor.py
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union, List
import math

import torch
import torch.nn.functional as F

from mmengine.logging import MMLogger
from mmengine.dist import get_rank

from mmseg.registry import MODELS

# SegDataPreProcessor is the baseline we want to match when morph is disabled
try:
    # mmseg >= 1.x
    from mmseg.models.data_preprocessor import SegDataPreProcessor  # type: ignore
except Exception:
    # fallback (older layouts)
    from mmseg.models.data_preprocessor.seg_data_preprocessor import SegDataPreProcessor  # type: ignore


def _as_4d(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is (N,C,H,W). Accepts (C,H,W) or (N,C,H,W)."""
    if x.dim() == 3:
        return x.unsqueeze(0)
    if x.dim() == 4:
        return x
    raise ValueError(f'Expected 3D/4D tensor, got shape={tuple(x.shape)}')


def _safe_min_max_mean(x: torch.Tensor) -> Tuple[float, float, float]:
    x = x.detach()
    return float(x.amin().item()), float(x.amax().item()), float(x.mean().item())


def _pad_right_bottom(img: torch.Tensor, target_hw: Tuple[int, int], pad_val: float) -> torch.Tensor:
    """Pad only on right & bottom to reach target (H,W). img: (C,H,W)."""
    h, w = int(img.shape[-2]), int(img.shape[-1])
    th, tw = int(target_hw[0]), int(target_hw[1])
    pad_h = th - h
    pad_w = tw - w
    if pad_h < 0 or pad_w < 0:
        # Should not happen if RGB padding defines target size; still guard.
        return img[..., :th, :tw]
    if pad_h == 0 and pad_w == 0:
        return img
    return F.pad(img, (0, pad_w, 0, pad_h), value=float(pad_val))


def _broadcast_mean_std(
    mean: Union[float, Sequence[float]],
    std: Union[float, Sequence[float]],
    c: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(mean, (list, tuple)):
        if len(mean) != c:
            raise ValueError(f'morph_mean has len={len(mean)} but morph channels={c}')
        m = torch.tensor(mean, dtype=torch.float32, device=device).view(1, c, 1, 1)
    else:
        m = torch.tensor(float(mean), dtype=torch.float32, device=device).view(1, 1, 1, 1)

    if isinstance(std, (list, tuple)):
        if len(std) != c:
            raise ValueError(f'morph_std has len={len(std)} but morph channels={c}')
        s = torch.tensor(std, dtype=torch.float32, device=device).view(1, c, 1, 1)
    else:
        s = torch.tensor(float(std), dtype=torch.float32, device=device).view(1, 1, 1, 1)

    return m, s


@MODELS.register_module()
class MorphSegDataPreProcessor(SegDataPreProcessor):
    """
    SegDataPreProcessor-compatible preprocessor for input-fusion:
      - When morph is disabled: behave like vanilla SegDataPreProcessor.
      - When morph is enabled: apply vanilla preprocessing to RGB first,
        then pad/normalize morph separately and concat as extra channels.

    Expected input channel order from pipeline:
      - before bgr_to_rgb: [B,G,R, morph...]
      - morph range ~ [0,1] float; RGB range ~ [0,255] (uint8/float)
    """

    def __init__(
        self,
        # --- baseline args (keep same semantics as SegDataPreProcessor) ---
        bgr_to_rgb: bool = True,
        mean: Sequence[float] = (123.675, 116.28, 103.53),
        std: Sequence[float] = (58.395, 57.12, 57.375),
        pad_val: float = 0.0,
        seg_pad_val: int = 255,
        size: Optional[Tuple[int, int]] = None,
        size_divisor: Optional[int] = None,

        # --- compatibility alias: allow passing rgb_mean/rgb_std from configs ---
        rgb_mean: Optional[Sequence[float]] = None,
        rgb_std: Optional[Sequence[float]] = None,

        # --- morph controls ---
        morph_enabled: bool = True,
        morph_weight: float = 1.0,
        morph_mean: Union[float, Sequence[float]] = 0.5,
        morph_std: Union[float, Sequence[float]] = 0.5,
        morph_expected_channels: int = 5,

        # --- debug ---
        debug: bool = False,
        debug_interval: int = 50,
        debug_show_channel_means: int = 8,

        **kwargs,
    ):
        # Priority rule:
        #   if rgb_mean/rgb_std provided -> use them
        #   else fallback to mean/std
        use_mean = tuple(rgb_mean) if rgb_mean is not None else tuple(mean)
        use_std = tuple(rgb_std) if rgb_std is not None else tuple(std)

        super().__init__(
            bgr_to_rgb=bgr_to_rgb,
            mean=use_mean,
            std=use_std,
            pad_val=pad_val,
            seg_pad_val=seg_pad_val,
            size=size,
            size_divisor=size_divisor,
            **kwargs,
        )

        self.morph_enabled = bool(morph_enabled)
        self.morph_weight = float(morph_weight)
        self.morph_mean = morph_mean
        self.morph_std = morph_std
        self.morph_expected_channels = int(morph_expected_channels)

        self.debug = bool(debug)
        self.debug_interval = int(debug_interval)
        self.debug_show_channel_means = int(debug_show_channel_means)

        self._call_idx = 0
        self._logger = MMLogger.get_current_instance()

    def _split_rgb_morph(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """x: (C,H,W) or (N,C,H,W). Return (rgb, morph_or_None)."""
        x4 = _as_4d(x)
        if x4.shape[1] <= 3:
            return x4, None
        rgb = x4[:, :3, :, :]
        morph = x4[:, 3:, :, :]
        return rgb, morph

    def _stack_morph_like_rgb(
        self,
        morph_inputs: Union[torch.Tensor, List[torch.Tensor]],
        target_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Build morph batch tensor matching padded RGB size.
        morph_inputs: Tensor (N,Mc,H,W)/(Mc,H,W) or list[(Mc,H,W)].
        """
        if isinstance(morph_inputs, (list, tuple)):
            m_list: List[torch.Tensor] = []
            for m in morph_inputs:
                if not isinstance(m, torch.Tensor):
                    m = torch.as_tensor(m)
                if m.dim() == 4:
                    # (1,Mc,H,W) -> (Mc,H,W)
                    m = m.squeeze(0)
                m = m.float()
                m = _pad_right_bottom(m, target_hw, pad_val=self.pad_val)
                m_list.append(m)
            if len(m_list) == 0:
                return torch.zeros((0,), device=self.rgb_mean.device)  # unreachable in practice
            return torch.stack(m_list, dim=0)  # (N,Mc,H,W)

        # Tensor path
        m4 = _as_4d(morph_inputs).float()
        # pad each sample (right/bottom only)
        n = m4.shape[0]
        out = []
        for i in range(n):
            out.append(_pad_right_bottom(m4[i], target_hw, pad_val=self.pad_val))
        return torch.stack(out, dim=0)

    def forward(self, data: dict, training: bool = False) -> dict:
        self._call_idx += 1

        # Cast once (to device); keep original for morph extraction
        data = self.cast_data(data)
        inputs = data.get('inputs', None)
        if inputs is None:
            raise KeyError('data must contain key "inputs".')

        # If morph disabled: exact baseline behavior
        if not self.morph_enabled:
            return super().forward(data, training=training)

        # ---- separate RGB & morph from raw inputs ----
        if isinstance(inputs, (list, tuple)):
            rgb_list = []
            morph_list = []
            for x in inputs:
                if not isinstance(x, torch.Tensor):
                    x = torch.as_tensor(x)
                # x expected (C,H,W)
                if x.dim() == 4:
                    x = x.squeeze(0)
                if x.shape[0] <= 3:
                    rgb_list.append(x)
                    morph_list.append(None)
                else:
                    rgb_list.append(x[:3])
                    morph_list.append(x[3:])
            # If any sample has morph, we treat morph as enabled; missing morph -> zeros
            any_morph = any(m is not None for m in morph_list)
            morph_inputs = [m if m is not None else torch.zeros_like(rgb_list[i][:1]).repeat(0, 1, 1)  # placeholder
                            for i, m in enumerate(morph_list)]
        else:
            x = inputs if isinstance(inputs, torch.Tensor) else torch.as_tensor(inputs)
            rgb, morph = self._split_rgb_morph(x)
            rgb_list = rgb  # tensor
            morph_inputs = morph  # tensor or None
            any_morph = morph is not None

        # ---- run baseline preprocessor on RGB only ----
        data_rgb = dict(data)
        data_rgb['inputs'] = rgb_list
        out = super().forward(data_rgb, training=training)
        rgb_batch: torch.Tensor = out['inputs']  # (N,3,H,W) float32 normalized

        # ---- no morph channels present: error when morph is enabled ----
        if not any_morph:
            raise ValueError(
                "morph_enabled=True but inputs have only 3 channels. "
                "Please ensure pipeline concatenates morphology channels."
            )

        # ---- build morph batch aligned to padded RGB size ----
        th, tw = int(rgb_batch.shape[-2]), int(rgb_batch.shape[-1])
        target_hw = (th, tw)

        if isinstance(inputs, (list, tuple)):
            # build morph tensor; handle None by zero-fill (Mc known per-sample)
            # Here we assume morph_list elements are either (Mc,H,W) or None.
            # For None, we create zeros with Mc=0, then later we will detect and skip.
            # Practically, your pipeline should always provide morph when enabled.
            # We'll enforce: all samples must have same Mc and not None.
            m_list_clean = []
            mc = None
            for i, x in enumerate(inputs):
                if not isinstance(x, torch.Tensor):
                    x = torch.as_tensor(x)
                if x.dim() == 4:
                    x = x.squeeze(0)
                if x.shape[0] <= 3:
                    raise ValueError(
                        'morph_enabled=True but some samples have only 3 channels. '
                        'Please ensure pipeline concatenates morph for all samples.'
                    )
                m = x[3:]
                mc = int(m.shape[0]) if mc is None else mc
                if int(m.shape[0]) != mc:
                    raise ValueError(f'Inconsistent morph channels across batch: got {m.shape[0]} vs {mc}')
                m_list_clean.append(m)
            if mc is not None and mc != self.morph_expected_channels:
                raise ValueError(
                    f'Expected morph channels={self.morph_expected_channels} but got {mc}. '
                    'Check LoadMorphologyAndConcat configuration.'
                )
            morph_batch = self._stack_morph_like_rgb(m_list_clean, target_hw)
        else:
            # tensor path
            if morph_inputs is None or morph_inputs.numel() == 0:
                raise ValueError(
                    "morph_enabled=True but inputs have only 3 channels. "
                    "Please ensure pipeline concatenates morphology channels."
                )
            mc = int(morph_inputs.shape[1])
            if mc != self.morph_expected_channels:
                raise ValueError(
                    f'Expected morph channels={self.morph_expected_channels} but got {mc}. '
                    'Check LoadMorphologyAndConcat configuration.'
                )
            morph_batch = self._stack_morph_like_rgb(morph_inputs, target_hw)

        # ---- normalize morph ----
        if morph_batch.numel() > 0:
            mc = int(morph_batch.shape[1])
            m_mean, m_std = _broadcast_mean_std(self.morph_mean, self.morph_std, mc, morph_batch.device)
            morph_batch = (morph_batch - m_mean) / (m_std + 1e-6)

            # weight gate
            morph_batch = morph_batch * float(self.morph_weight)

            # concat
            out['inputs'] = torch.cat([rgb_batch, morph_batch], dim=1)

        # ---- debug (rank0 only) ----
        if self.debug and training and (self._call_idx % max(1, self.debug_interval) == 0) and get_rank() == 0:
            with torch.no_grad():
                xcat = out['inputs']
                self._logger.info(
                    f'[MorphDP] call={self._call_idx} '
                    f'morph_enabled={self.morph_enabled} morph_weight={self.morph_weight:.3f} '
                    f'inputs shape={tuple(xcat.shape)} dtype={xcat.dtype}'
                )
                rmn, rmx, rmean = _safe_min_max_mean(rgb_batch)
                self._logger.info(f'[MorphDP] RGB(norm) min/max/mean = {rmn:.4f}/{rmx:.4f}/{rmean:.4f}')

                if morph_batch.numel() > 0:
                    mmn, mmx, mmean = _safe_min_max_mean(morph_batch)
                    self._logger.info(f'[MorphDP] Morph(norm*weight) min/max/mean = {mmn:.4f}/{mmx:.4f}/{mmean:.4f}')

                    k = min(self.debug_show_channel_means, int(morph_batch.shape[1]))
                    ch_means = morph_batch.mean(dim=(0, 2, 3))[:k].detach().cpu().tolist()
                    ch_means_str = ', '.join([f'{v:.4f}' for v in ch_means])
                    self._logger.info(f'[MorphDP] Morph first {k} channel means: {ch_means_str}')

        return out
