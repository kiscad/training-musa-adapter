"""Synthetic NVIDIA-scale values for Megatron's architecture comparisons.

MUSA capability numbers are not NVIDIA compute capabilities. The default 8.3
passes Megatron's >=8 grouped-GEMM gate and stays below its >=10 architecture
checks. It neither describes real hardware nor proves that a gated kernel is
supported. The claim is fixed; exposing new NVIDIA paths requires separate validation.
The torch.cuda override is process-wide once Megatron triggers it; undo restores
it only while we still own the attribute. Real MUSA capability queries should
use the backend's native API.
"""

from __future__ import annotations

import logging
from typing import Any

from ..._engine import AttrPatch, HookPatch
from ...backends import musa_available

__all__ = ["PATCHES", "arch_tuple", "arch_major"]

logger = logging.getLogger("training_musa_adaptor")

_DEFAULT_ARCH = (8, 3)
_MISSING = object()
_capability_override: tuple[Any, Any, Any] | None = None


def arch_tuple() -> tuple[int, int]:
    """The verified capability claim for this stack is fixed at the default. A different claim is a new
    patch revision with its own evidence, not a deployment knob."""
    return _DEFAULT_ARCH


def arch_major() -> int:
    return arch_tuple()[0]


def _install_torch_capability() -> bool | None:
    global _capability_override
    if _capability_override is not None or not musa_available():
        return False

    import torch

    capability = arch_tuple()

    def get_device_capability(device=None):
        return capability

    owner = torch.cuda
    # Some compatibility namespaces resolve/cache attributes in __getattr__.
    # Journal the direct attribute only; a lookup would itself mutate the owner.
    original = vars(owner).get("get_device_capability", _MISSING)
    owner.get_device_capability = get_device_capability
    _capability_override = (owner, original, get_device_capability)
    return None  # applied; the hook contract distinguishes decline (False)


def _uninstall_torch_capability() -> None:
    global _capability_override
    if _capability_override is None:
        return
    owner, original, replacement = _capability_override
    if vars(owner).get("get_device_capability", _MISSING) is replacement:
        if original is _MISSING:
            delattr(owner, "get_device_capability")
        else:
            owner.get_device_capability = original
    _capability_override = None


def _replace_arch_version(original: Any) -> Any:
    if not musa_available():
        return None
    major = arch_major()

    def get_device_arch_version():
        return major

    return get_device_arch_version


PATCHES = (
    HookPatch(
        id="torch.cuda.device-capability.nvidia-scale",
        trigger="megatron.core.parallel_state",
        run=_install_torch_capability,
        undo=_uninstall_torch_capability,
        # torch-level policy; the megatron consumer (the NVIDIA capability
        # gate in megatron/training/arguments.py, current package layout)
        # exists from core_v0.6.0 and is unchanged through core_v0.19.0, the
        # newest release line in the checkout.
        version_gates=("megatron-core >=0.6,<0.20",),
        rationale=(
            "Megatron arguments.py compares torch.cuda.get_device_capability "
            "against NVIDIA's >=8 grouped-GEMM threshold; MUSA's native numbering "
            "is not comparable. A numeric gate is not a backend kernel probe."
        ),
        strategy=(
            "On Megatron import with a live MUSA runtime substitute the fixed "
            "NVIDIA-scale pair (8.3) on torch.cuda only; record ownership for reversible undo. "
            "This is a process-wide policy, not a claim of Ampere compatibility."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/training/arguments.py; "
            "pytorch/pytorch torch/cuda/__init__.py"
        ),
        remove_when=(
            "Review capability consumers on every Megatron and torch_musa/backend "
            "upgrade; remove when Megatron uses backend feature probes. Before "
            "removal test grouped GEMM and architecture-gated paths on target hardware."
        ),
    ),
    AttrPatch(
        id="megatron.training.get-device-arch-version.nvidia-scale",
        rebind_prefixes=("megatron",),
        target="megatron.training.utils:get_device_arch_version",
        # The helper exists from core_v0.12.0. core_v0.18.0 turns
        # megatron/training/utils into a package whose __init__ re-exports it
        # with the same contract, so the target still resolves there
        # (verified through core_v0.19.0, the newest release line).
        version_gates=("megatron-core >=0.12,<0.20",),
        replace=_replace_arch_version,
        rationale=(
            "The upstream helper reads CUDA device properties.major and documents "
            "NVIDIA architecture numbers. MUSA's native major would be interpreted "
            "as a NVIDIA generation in connection-limit and stream-priority checks."
        ),
        strategy=(
            "Return the same synthetic major as the capability hook without "
            "constructing or querying a CUDA device. Default 8 stays below 10; "
            "this does not bypass every NVIDIA-specific code path."
        ),
        upstream="NVIDIA/Megatron-LM megatron/training/utils.py; arguments.py",
        remove_when=(
            "Review on Megatron and backend upgrades; remove with the capability "
            "hook when architecture checks become backend-aware, after TP/CP/FSDP "
            "connection-limit and priority-stream initialization tests."
        ),
    ),
)
