"""Megatron Core GDN patch: TileLang dispatch of the chunked gated delta rule.

The dispatcher lives in ``ops/gated_delta_rule.py`` (shared with the
mcore-bridge call site).  Disable with
``TRAINING_MUSA_ADAPTOR_DISABLE=megatron.ssm.gated-delta-rule.tilelang``.
"""

from __future__ import annotations

from typing import Any

from ..._engine import AttrPatch
from ...ops.gated_delta_rule import make_tilelang_dispatcher

__all__ = ["PATCHES"]

_CORE_GDN = "megatron.core.ssm.gated_delta_net"


def _replace(original: Any) -> Any:
    return make_tilelang_dispatcher(original)


PATCHES = (
    AttrPatch(
        id="megatron.ssm.gated-delta-rule.tilelang",
        target=f"{_CORE_GDN}:chunk_gated_delta_rule",
        rebind_prefixes=("megatron",),
        # megatron/core/ssm/gated_delta_net.py:chunk_gated_delta_rule exists
        # from core_v0.16.0. core_v0.19.0 turns it into a package whose
        # __init__ re-exports the same flash-linear-attention binding, so the
        # target still resolves there (verified through core_v0.19.0, the
        # newest release line in the checkout).
        version_gates=("megatron-core >=0.16,<0.20",),
        replace=_replace,
        rationale=(
            "Megatron's GatedDeltaNet computes the chunked gated delta rule with "
            "flash-linear-attention's Triton kernels. On MUSA that Triton path is "
            "the slowest part of Qwen3.5-style hybrid models (24 of 32 layers are "
            "GDN): torch-kernels' TileLang implementation of the identical "
            "operator, signature and [B,T,H,D] layout measures 8-10x end to end "
            "against fla 0.5.2 on MTT S5000 (torch_kernels GDN benchmark)."
        ),
        strategy=(
            "Wrap Megatron's own FLA binding, not the fla package: calls whose "
            "tensors, dtypes and shapes the TileLang kernels document as "
            "supported (MUSA, bf16 q/k/v/beta, fp32 g, gdn_dense_supported / "
            "gdn_varlen_supported) are dispatched to "
            "torch_kernels.attention.gated_delta_net (tilelang backend), which "
            "handles chunk padding, GVA head expansion and autograd itself. "
            "Every other call -- CPU/CUDA tensors, fp16/fp32 activations, short "
            "sequences, fla-only keywords such as use_beta_sigmoid_in_kernel or "
            "cp_context -- is forwarded to FLA with the arguments untouched, so "
            "FLA stays the correctness reference and the fallback. The dispatch "
            "checks shapes per call from tensor metadata only; packed calls "
            "read cu_seqlens once (small sync) to validate the per-sequence "
            "two-chunk precondition. The TileLang kernels JIT-compile on first "
            "use per head count and dense/unpadded specialization (minutes, "
            "cached in ~/.tilelang); on multi-rank runs pre-warm the cache once "
            "in a single process for the intended shapes before workers start; "
            "every rank compiling the "
            "same kernels into the shared cache concurrently has crashed runs "
            "(device error / SIGABRT). The TileLang stack is version-bound: "
            "tilelang-musa and torch-kernels upgrade only as a matched set with "
            "the MUSA stack, and a stack change also invalidates the "
            "~/.tilelang kernel cache (drop it and re-warm). "
            "TRAINING_MUSA_ADAPTOR_DISABLE=megatron.ssm.gated-delta-rule.tilelang "
            "restores upstream's FLA binding entirely."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/ssm/gated_delta_net.py:chunk_gated_delta_rule",
        remove_when=(
            "Remove when flash-linear-attention ships MUSA kernels matching the "
            "TileLang performance, or when Megatron selects a kernel through an "
            "extension seam this package can implement more directly; disable "
            "this patch id and compare the 3-iteration ms-swift Qwen3.5 run and "
            "the gdn smoke before deleting."
        ),
    ),
)
