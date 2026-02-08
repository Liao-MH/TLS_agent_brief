from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


class TLSKitError(Exception):
    """Base exception for toolkit."""


class WSIReadError(TLSKitError):
    """WSI open/read failure."""


class ResourceError(TLSKitError):
    """OOM, disk full, etc."""


class InferenceError(TLSKitError):
    """Model inference failure."""


class ArtifactError(TLSKitError):
    """Missing/invalid intermediate artifacts."""


@dataclass
class ClassifiedError:
    """
    A lightweight error taxonomy for Agent to provide suggestions.

    category:
      - io
      - resource
      - dependency
      - data_quality
      - logic
      - unknown
    """
    category: str
    message: str
    suggestion: str
    detail: Optional[str] = None
