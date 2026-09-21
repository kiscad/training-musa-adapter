"""Device-backend helpers shared by patches.

Everything in this subpackage changes the *runtime* (``torch``), not a
framework.  Helpers are idempotent and record their ownership/cleanup with
the engine; ``musa_available`` never imports torchada just to probe.
"""

from __future__ import annotations

from . import torch_cuda

__all__ = ["torch_cuda", "musa_available"]


def musa_available() -> bool:
    """Query the current runtime without importing torchada or caching state."""
    import torch

    musa = getattr(torch, "musa", None)
    available = getattr(musa, "is_available", None)
    return callable(available) and bool(available())
