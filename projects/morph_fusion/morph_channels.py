# projects/morph_fusion/morph_channels.py

from __future__ import annotations

from typing import Dict, List, Sequence


# 固定 morphology 文件内部的通道顺序（强烈建议与你的 morphology 生成脚本保持一致）
# 你当前消融矩阵用到的最小集合：H / GradMag / LBP / Gabor_S1 / Gabor_S2
MORPH_CHANNEL_ORDER: List[str] = [
    'H',
    'GradMag',
    'LBP',
    'Gabor_S1',
    'Gabor_S2',
]

# 允许一些常见别名（如果你 morphology 文件里用别的命名，可在这里扩展）
_ALIASES: Dict[str, str] = {
    'hematoxylin': 'H',
    'h_channel': 'H',
    'gradmag': 'GradMag',
    'grad_mag': 'GradMag',
    'lbp': 'LBP',
    'gabor1': 'Gabor_S1',
    'gabor_s1': 'Gabor_S1',
    'gabor2': 'Gabor_S2',
    'gabor_s2': 'Gabor_S2',
}

# 标准名 -> index
MORPH_CHANNEL_TO_INDEX: Dict[str, int] = {name: i for i, name in enumerate(MORPH_CHANNEL_ORDER)}


def _canonicalize(name: str) -> str:
    key = name.strip()
    key_lower = key.lower()
    if key in MORPH_CHANNEL_TO_INDEX:
        return key
    if key_lower in _ALIASES:
        return _ALIASES[key_lower]
    # 也允许用户直接写 “GradMag” 这种大小写不同的形式
    for std in MORPH_CHANNEL_TO_INDEX.keys():
        if std.lower() == key_lower:
            return std
    raise KeyError(
        f'Unknown morph channel name: {name!r}. '
        f'Allowed: {list(MORPH_CHANNEL_TO_INDEX.keys())}, plus aliases: {list(_ALIASES.keys())}'
    )


def resolve_morph_channel_indices(channel_names: Sequence[str]) -> List[int]:
    """Map channel_names to indices based on MORPH_CHANNEL_ORDER."""
    if not channel_names:
        raise ValueError('channel_names is empty; please provide at least one morph channel.')

    indices: List[int] = []
    seen = set()
    for n in channel_names:
        cn = _canonicalize(n)
        idx = MORPH_CHANNEL_TO_INDEX[cn]
        if idx in seen:
            continue
        seen.add(idx)
        indices.append(idx)
    return indices

