"""Fused-softmax availability that tells the truth about the CUDA extension.

``FusedScaleMaskSoftmax.is_kernel_available`` reaches ``get_batch_per_block``,
which imports the ``scaled_masked_softmax_cuda`` extension at call time. On the
MUSA stack that module does not exist, so instead of the intended torch
fallback the model dies with ``ModuleNotFoundError``. Decline the fused kernel
before that import and the upstream torch path runs unchanged.
"""

from __future__ import annotations

import functools
from typing import Any

from ..._engine import AttrPatch

__all__ = ["PATCHES"]


_SOFTMAX_EXTENSION = "scaled_masked_softmax_cuda"


def _softmax_kernel_available(original: Any) -> Any:
    """Return False when the fused CUDA extension is absent."""

    import importlib.util

    if importlib.util.find_spec(_SOFTMAX_EXTENSION) is not None:
        return None  # Keep the upstream probe when its extension exists.

    @functools.wraps(original)
    def is_kernel_available(self, mask, b, np, sq, sk):
        return False

    return is_kernel_available


PATCHES = (
    AttrPatch(
        id="megatron.softmax.kernel-availability.musa",
        rebind_prefixes=("megatron",),
        target=(
            "megatron.core.fusions.fused_softmax:FusedScaleMaskSoftmax."
            "is_kernel_available"
        ),
        # FusedScaleMaskSoftmax.is_kernel_available and the get_batch_per_block
        # probe exist unchanged from core_v0.4.0 through core_v0.19.0, the
        # newest release line in the checkout.
        version_gates=("megatron-core >=0.4,<0.20",),
        replace=_softmax_kernel_available,
        rationale=(
            "Upcycling with the local spec constructs FusedScaleMaskSoftmax; its "
            "is_kernel_available probe reaches get_batch_per_block, which does a "
            "bare `import scaled_masked_softmax_cuda` at call time. The MUSA "
            "stack has no such extension, so the probe itself raises "
            "ModuleNotFoundError instead of selecting the torch fallback "
            "(test_upcycling_Local[tp_ep0-1-False-False-False])."
        ),
        strategy=(
            "At installation, check whether the extension is importable "
            "with importlib.util.find_spec (which locates without executing the "
            "module) and return False when it is absent, so upstream's own "
            "forward_torch_softmax path runs with its scale/mask/fp32-softmax "
            "semantics. When the extension exists, the original probe decides."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/core/fusions/fused_softmax.py:"
            "FusedScaleMaskSoftmax.is_kernel_available"
        ),
        remove_when=(
            "Remove when a MUSA build of scaled_masked_softmax_cuda exists or "
            "upstream probes the extension's availability instead of importing "
            "it inside the kernel check; re-run the upcycling local case and "
            "fusions/test_torch_softmax.py with this patch disabled."
        ),
    ),
)
