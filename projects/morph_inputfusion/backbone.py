#/lustre1/g/path_dwhho/new_LMH/mmsegmentation/projects/morph_inputfusion/backbone.py


from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from mmseg.registry import MODELS
from mmseg.models.backbones.swin import SwinTransformer


@MODELS.register_module()
class MorphSwinTransformer(SwinTransformer):
    def __init__(self, morph: Optional[Dict] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        morph = morph or {}
        self.morph_enabled = bool(morph.get('enabled', False))
        self.morph_num_channels = int(morph.get('num_channels', 0))
        self.project_to_rgb = bool(morph.get('project_to_rgb', True))

        if self.morph_enabled:
            if self.morph_num_channels <= 0:
                raise ValueError('morph.enabled=True requires morph.num_channels > 0')
            if not self.project_to_rgb:
                raise NotImplementedError('Only project_to_rgb=True is supported.')
            self.morph_proj = nn.Conv2d(3 + self.morph_num_channels, 3, kernel_size=1, bias=False)
        else:
            self.morph_proj = None

    def forward(self, x: torch.Tensor):
        if self.morph_enabled:
            exp = 3 + self.morph_num_channels
            if x.size(1) != exp:
                raise ValueError(f'Expected {exp} channels (RGB+morph) but got {x.size(1)}')
            x = self.morph_proj(x)
        else:
            if x.size(1) != 3:
                raise ValueError(f'Morph disabled but input has {x.size(1)} channels; disable morph loader in pipeline.')
        return super().forward(x)
