"""FP32 gradient-norm dtype for DeepSpeed ZeRO on MUSA.

DeepSpeed's ZeRO grad-norm reduction accumulates per-tensor norms in fp64
(``deepspeed.runtime.zero.utils.get_norm_dtype`` returns ``torch.double``
whenever ``get_accelerator().is_fp64_supported()`` is true).  The CUDA
accelerator reports ``is_fp64_supported() == True`` on this stack because
``torch.float64`` tensors exist, but torch_musa does not implement the
``norm`` kernel for fp64 -- ZeRO-3's very first optimizer step dies in
``get_grad_norm_direct`` with ``RuntimeError: Norm only supports
Float/Half/BFloat16 input, but now it is Double``.

DeepSpeed's own contract already anticipates devices without usable fp64:
the docstring says gradient norms fall back to fp32 on such devices, and the
MPS accelerator answers ``is_fp64_supported() == False`` for exactly this
reason.  This patch reports the same truth for MUSA: torch_musa tensors
exist in fp64, but the norm kernel the reduction needs does not.

Scope: only the norm-dtype selection.  All seven call sites
(``zero/stage3.py`` and ``zero/stage_1_and_2.py`` norm groups, grad-norm
reductions) read ``get_norm_dtype()`` at call time, so one patch covers the
whole ZeRO family.  Gradient norms are reduced in fp32 instead of fp64 --
the same precision Megatron-LM and PyTorch's own clip_grad_norm_ use; the
norm feeds logging and clipping, not parameter updates.  Non-MUSA processes
keep the original fp64 behavior.

Boundary note: ``zero/stage3.py`` and ``zero/stage_1_and_2.py`` bind
``get_norm_dtype`` with ``from deepspeed.runtime.zero.utils import ...`` at
their import time.  The AttrPatch applies at the ``zero.utils`` execution
boundary, which precedes both, so the patched probe is what they bind.

Version gates: ``get_norm_dtype`` first appears in 0.19.0 (sdists 0.17.2 /
0.18.9 do not have it; 0.19.0-0.19.7 carry the same signature and
``is_fp64_supported()``-based logic -- verified 0.19.7 sources).  0.20 does
not exist yet; the upper bound is a conservative verification boundary.
"""

from __future__ import annotations

import functools
from typing import Any

from .._engine import AttrPatch

__all__ = ["PATCHES"]


def replace_get_norm_dtype(original: Any) -> Any:
    """Report fp32 norm dtype on MUSA; defer to DeepSpeed everywhere else.

    DeepSpeed selects fp64 for gradient-norm accumulation when the
    accelerator claims fp64 support.  torch_musa exposes the fp64 *dtype*
    but lacks the *norm kernel* DeepSpeed's reduction calls, so on a MUSA
    device this returns ``torch.float`` (DeepSpeed's documented fallback for
    devices without usable fp64).  Every other device -- and any future
    torch_musa that grows an fp64 norm kernel, should DeepSpeed's probe
    change -- keeps the original answer.
    """

    @functools.wraps(original)
    def get_norm_dtype() -> Any:
        import torch

        dtype = original()
        musa_available = hasattr(torch, "musa") and torch.musa.is_available()
        if dtype is torch.double and musa_available:
            return torch.float
        return dtype

    return get_norm_dtype


PATCHES = (
    AttrPatch(
        id="deepspeed.zero.grad-norm.fp32",
        target="deepspeed.runtime.zero.utils:get_norm_dtype",
        replace=replace_get_norm_dtype,
        version_gates=("deepspeed >=0.19,<0.20",),
        rationale=(
            "ZeRO-3's first optimizer step dies on MUSA with 'Norm only "
            "supports Float/Half/BFloat16 input, but now it is Double': "
            "get_norm_dtype returns torch.double because the CUDA "
            "accelerator's is_fp64_supported() only checks that the dtype "
            "exists, while torch_musa implements no fp64 norm kernel. "
            "DeepSpeed's own contract routes devices without usable fp64 "
            "(MPS) to fp32; MUSA belongs in that class."
        ),
        strategy=(
            "Wrap get_norm_dtype only: on a MUSA device (torch.musa "
            "available) a torch.double answer becomes torch.float, so all "
            "ZeRO grad-norm reductions run in fp32 -- the same precision "
            "Megatron-LM and PyTorch clip_grad_norm_ use; the norm feeds "
            "clipping/logging, not parameter updates. Every other device "
            "and every other dtype keeps the original answer, and torch is "
            "imported inside the factory. Boundary: the patch applies at "
            "the deepspeed.runtime.zero.utils exec boundary, before "
            "stage3.py / stage_1_and_2.py bind the name by-value. "
            "TRAINING_MUSA_ADAPTOR_DISABLE=deepspeed.zero.grad-norm.fp32 "
            "restores the fp64 selection (and the crash)."
        ),
        upstream=(
            "deepspeedai/DeepSpeed deepspeed/runtime/zero/utils.py:"
            "get_norm_dtype (0.19.x); callers in zero/stage3.py:"
            "get_grad_norm_direct/_get_norm_groups and "
            "zero/stage_1_and_2.py; accelerator/mps_accelerator.py:"
            "is_fp64_supported precedent"
        ),
        remove_when=(
            "Remove when torch_musa implements the fp64 norm kernel or "
            "DeepSpeed's accelerator probes actual kernel support instead "
            "of dtype existence; verify by running a ZeRO-3 step with this "
            "patch disabled."
        ),
    ),
)
