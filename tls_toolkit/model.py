from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import importlib

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
    """
    Load mmsegmentation model and return bundle with metadata.

    Import mmseg lazily so the toolkit can still be imported in environments
    where mmseg is not installed (e.g., documentation tooling).
    """
    try:
        from mmseg.apis import init_model  # type: ignore
        from mmengine.config import Config  # type: ignore
    except Exception as e:
        raise InferenceError(f"mmseg is not available: {e}")

    try:
        if config_file:
            cfg = Config.fromfile(config_file)
            custom_imports = cfg.get("custom_imports", None)
            if isinstance(custom_imports, dict):
                allow_failed = bool(custom_imports.get("allow_failed_imports", False))
                for mod in custom_imports.get("imports", []):
                    try:
                        importlib.import_module(mod)
                    except Exception:
                        if not allow_failed:
                            raise
        model = init_model(config_file, checkpoint, device=device)
        model.eval()
    except Exception as e:
        raise InferenceError(f"Failed to init mmseg model: {e}")

    meta = {
        "config_file": config_file,
        "config_sha256": sha256_file(config_file) if config_file else None,
        "checkpoint": checkpoint,
        "checkpoint_sha256": sha256_file(checkpoint) if checkpoint else None,
        "device": device,
        "torch_version": torch.__version__,
    }
    return ModelBundle(model=model, meta=meta)
