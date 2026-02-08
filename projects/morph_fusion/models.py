# projects/morph_fusion/models.py

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS


@MODELS.register_module()
class InputFusionBackbone(BaseModule):
    """
    Scheme-1 Input Fusion:
      - Input: x with shape (B, 3+M, H, W) or (B, 3, H, W)
      - If morph.enabled=False:
            use x[:, :3] as RGB, forward to inner backbone (baseline-identical).
      - If morph.enabled=True:
            concat already done in data pipeline, so x is (B, 3+M, H, W)
            apply 1x1 conv projection: (3+M)->3, then forward to inner backbone.

    Key requirement:
      - When morph is disabled, forward path is identical to baseline (no extra conv).
      - When enabled, projection conv is initialized as:
            output = identity(RGB) + 0*morph
        so training starts from baseline behavior.
    """

    def __init__(
        self,
        backbone: Dict[str, Any],
        morph: Optional[Dict[str, Any]] = None,
        init_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.morph_cfg = morph or {}
        self.morph_enabled: bool = bool(self.morph_cfg.get('enabled', False))
        self.num_morph_channels: int = int(self.morph_cfg.get('num_morph_channels', 0))

        # build inner backbone
        self.backbone = MODELS.build(backbone)

        # fusion projection (only used if enabled)
        self.proj: Optional[nn.Conv2d] = None
        if self.morph_enabled:
            if self.num_morph_channels <= 0:
                raise ValueError('morph.enabled=True but num_morph_channels <= 0.')
            in_ch = 3 + self.num_morph_channels
            self.proj = nn.Conv2d(in_channels=in_ch, out_channels=3, kernel_size=1, bias=True)
            self._init_proj_identity(self.proj, num_morph=self.num_morph_channels)

    @staticmethod
    def _init_proj_identity(conv: nn.Conv2d, num_morph: int) -> None:
        # weight shape: (3, 3+M, 1, 1)
        with torch.no_grad():
            conv.weight.zero_()
            conv.bias.zero_()
            # identity for RGB part
            for c in range(3):
                conv.weight[c, c, 0, 0] = 1.0
            # morph part stays 0 -> initially no effect

    def forward(self, x: torch.Tensor) -> Any:
        if x.dim() != 4:
            raise ValueError(f'Expected input x as (B,C,H,W), got shape={tuple(x.shape)}')

        if not self.morph_enabled:
            # baseline path: strictly use first 3 channels
            if x.size(1) > 3:
                x = x[:, :3, :, :]
            return self.backbone(x)

        # enabled path
        expected_c = 3 + self.num_morph_channels
        if x.size(1) != expected_c:
            raise ValueError(
                f'morph.enabled=True expects C={expected_c} (3+num_morph_channels), '
                f'but got C={x.size(1)}. Check your pipeline concat and channel_names.'
            )

        assert self.proj is not None
        x = self.proj(x)
        return self.backbone(x)

    def init_weights(self) -> None:
        # initialize inner backbone weights
        if hasattr(self.backbone, 'init_weights'):
            self.backbone.init_weights()

        # proj already identity-initialized in __init__; nothing more required.

    def __getattr__(self, name: str) -> Any:
        """
        Delegate unknown attributes to inner backbone.
        This helps neck/head read properties like out_channels, embed_dims, etc.
        """
        # IMPORTANT: avoid infinite recursion
        if name in {'backbone', 'morph_cfg', 'morph_enabled', 'num_morph_channels', 'proj'}:
            return super().__getattribute__(name)
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(super().__getattribute__('backbone'), name)

