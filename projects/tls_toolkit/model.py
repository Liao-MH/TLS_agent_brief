# tls_toolkit/model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import torch

from .utils import sha256_file
from .errors import InferenceError


@dataclass
class ModelBundle:
    model: Any
    meta: Dict[str, Any]


def load_mmseg_model(
    config_file: str,
    checkpoint: str,
    device: str,
) -> ModelBundle:
    try:
        from mmseg.apis import init_model  # type: ignore
    except Exception as e:
        raise InferenceError(f"mmseg is not available: {e}")

    try:
        model = init_model(config_file, checkpoint, device=device)
        model.eval()
        if getattr(model, "cfg", None) is None:
            raise InferenceError("init_model succeeded but model.cfg is missing; cannot reuse test_pipeline.")
    except Exception as e:
        raise InferenceError(f"Failed to init mmseg model: {e}")

    meta = {
        "config_file": config_file,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
        "device": device,
        "torch_version": torch.__version__,
    }
    return ModelBundle(model=model, meta=meta)
