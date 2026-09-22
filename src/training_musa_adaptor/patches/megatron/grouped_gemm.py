"""Device-side grouped GEMM reference for MUSA.

The upstream MoE grouped-GEMM path requires the fanshiqing ``grouped_gemm``
CUDA extension, which has no MUSA build. This module provides a reference
``ops.gmm`` built from per-expert ``torch.matmul`` so ``GroupedMLP`` keeps its
weights, checkpoint keys and sharding exactly as upstream. It is a performance
fallback, not a second model implementation: one GEMM per local expert.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from ..._engine import AttrPatch

__all__ = ["PATCHES"]


logger = logging.getLogger("training_musa_adaptor")

_UTIL = "megatron.core.transformer.moe.grouped_gemm_util"


def _grouped_gemm_module():
    import sys

    return sys.modules.get(_UTIL)


def _vendor_grouped_gemm_missing() -> bool:
    module = _grouped_gemm_module()
    return module is not None and getattr(module, "grouped_gemm", None) is None


def gmm(a, b, tokens_per_expert, trans_b=False):
    """Grouped matmul: ``y[e] = a[e] @ b[e]`` (or ``b[e].T`` with trans_b).

    ``a`` is [num_tokens, K] with tokens already grouped by expert, ``b`` is
    [num_experts, K, N] (trans_b=False), and ``tokens_per_expert`` gives each
    expert's row count. Only the token counts cross to the CPU; every matmul
    stays on the activations' device and autograd flows through the split and
    the concatenation, so empty experts still receive zero gradients instead
    of losing the edge.
    """
    import torch

    if a.ndim != 2 or b.ndim != 3:
        raise ValueError(
            "gmm expects a [tokens, K] and b [experts, K, N] (or [experts, N, K])"
        )
    if tokens_per_expert.ndim != 1 or tokens_per_expert.numel() != b.size(0):
        raise ValueError("tokens_per_expert must contain one count per expert")
    counts = tokens_per_expert.tolist()
    if any(not isinstance(count, int) or count < 0 for count in counts):
        raise ValueError("tokens_per_expert must contain nonnegative integer counts")
    if sum(counts) != a.size(0):
        raise ValueError(
            f"tokens_per_expert sums to {sum(counts)} but a has {a.size(0)} rows"
        )
    if trans_b:
        b = b.transpose(-2, -1)
    if not counts:
        # Even the zero-expert case retains both autograd edges.
        return torch.matmul(a, b.sum(dim=0))
    outputs = [
        torch.matmul(chunk, weight)
        for chunk, weight in zip(a.split(counts), b, strict=True)
    ]
    return torch.cat(outputs, dim=0)


class _GroupedGemmOps:
    """Minimal stand-in for ``grouped_gemm.ops``."""

    gmm = staticmethod(gmm)


def _grouped_gemm_ops(original: Any) -> Any:
    if not _vendor_grouped_gemm_missing():
        return None
    return _GroupedGemmOps()


def _grouped_gemm_is_available(original: Any) -> Any:
    if not _vendor_grouped_gemm_missing():
        return None

    def grouped_gemm_is_available() -> bool:
        return True

    return grouped_gemm_is_available


def _assert_grouped_gemm_is_available(original: Any) -> Any:
    if not _vendor_grouped_gemm_missing():
        return None

    @functools.wraps(original)
    def assert_grouped_gemm_is_available() -> None:

        module = _grouped_gemm_module()
        check = getattr(module, "grouped_gemm_is_available", None)
        if check is not None and not check():
            raise AssertionError(
                "Grouped GEMM is not available. Please run "
                "`pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4`."
            )

    return assert_grouped_gemm_is_available


def _te_grouped_wgrad(original):
    """Keep TE's saved/offloaded tensors; replace only plain NT GEMM+bgrad."""
    import inspect

    from ...backends import musa_available

    if not musa_available():
        return None
    signature = inspect.signature(original)

    @functools.wraps(original)
    def grouped(*args, **kwargs):
        import torch

        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        v = bound.arguments
        a, b, outputs = v["A"], v["B"], v["out"]
        if not (
            v["layout"] == "NT"
            and v["grad"]
            and not v["gelu"]
            and not v["single_output"]
            and v["D_dtype"] is None
            and a
            and len(a) == len(b) == len(outputs)
            and all(
                type(t) is torch.Tensor
                and t.device.type == "musa"
                and t.dtype in (torch.float16, torch.bfloat16, torch.float32)
                for t in (*a, *b, *outputs)
            )
        ):
            return original(*args, **kwargs)
        if not v["use_bias"]:
            return original(*args, **kwargs)
        # MuDNN NT GEMM supports wgrad, but not fused bgrad. Keep its
        # workspace/accumulation semantics and reduce bias separately.
        v["use_bias"] = False
        result, _, gelu_inputs = original(*bound.args, **bound.kwargs)
        return result, [dy.sum(0) for dy in b], gelu_inputs

    return grouped


PATCHES = (
    AttrPatch(
        id="transformer_engine.grouped-gemm.wgrad-reference",
        target="transformer_engine.pytorch.cpp_extensions.gemm:general_grouped_gemm",
        rebind_prefixes=("transformer_engine",),
        # TE-side target; its megatron-side consumers (GroupedMLP/TEGroupedLinear
        # offload flows) are present through core_v0.19.0, the newest release
        # line in the checkout. MT-TE 2.0 is the fork with the broken grouped
        # NT+bgrad kernel.
        version_gates=(
            "megatron-core >=0.16,<0.20",
            "transformer_engine >=2.0,<2.1",
        ),
        replace=_te_grouped_wgrad,
        rationale="MT-TE plain grouped NT weight-gradient GEMM with bias fails RunLt in offloading tests.",
        strategy=(
            "Use native grouped GEMM without fused bgrad and separate bias reduction for MUSA NT gradients, "
            "writing the original output buffers and honoring accumulation. TE retains "
            "its autograd and activation offload protocol; FP8 and other layouts stay native. "
            "This adds one bias reduction per expert."
        ),
        upstream="TransformerEngine pytorch/cpp_extensions/gemm.py:general_grouped_gemm",
        remove_when="Remove when native grouped NT+bgrad passes offloading tests and accumulation checks.",
    ),
    AttrPatch(
        id="megatron.moe.grouped-gemm.torch-ops",
        rebind_prefixes=("megatron",),
        target=f"{_UTIL}:ops",
        # megatron/core/transformer/moe/grouped_gemm_util.py (ops,
        # grouped_gemm_is_available, assert_grouped_gemm_is_available) exists
        # from core_v0.5.0 and is removed in core_v0.17.0, which reworks
        # grouped-GEMM backend selection (inference_grouped_gemm_backend):
        # there is nothing to patch beyond that line.
        version_gates=("megatron-core >=0.5,<0.17",),
        replace=_grouped_gemm_ops,
        rationale=(
            "GroupedMLP builds through gg.assert_grouped_gemm_is_available() and "
            "runs gg.ops.gmm, but the fanshiqing grouped_gemm extension is a CUDA "
            "build with no MUSA port, so every grouped-expert construction fails "
            "with 'Grouped GEMM is not available' (72 upstream cases in "
            "dist_checkpointing/models/test_moe_experts.py)."
        ),
        strategy=(
            "When the vendor package is absent, provide a minimal ops namespace "
            "whose gmm splits the activations by tokens_per_expert and runs one "
            "torch.matmul per local expert on the device, honouring trans_b and "
            "concatenating in expert order. Only the token counts cross to the "
            "CPU; autograd flows through the split/cat so empty experts keep "
            "zero (not None) weight gradients, and weight1/weight2 layouts, "
            "checkpoint keys and expert sharding are untouched. Declines when a "
            "real grouped_gemm is importable."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/core/transformer/moe/grouped_gemm_util.py; "
            "fanshiqing/grouped_gemm ops.gmm"
        ),
        remove_when=(
            "Remove when a validated MUSA grouped_gemm (vendor or community) is "
            "installable: disable this patch id and re-run "
            "dist_checkpointing/models/test_moe_experts.py plus a GroupedMLP "
            "forward/backward comparison; delete only if the vendor path passes."
        ),
    ),
    AttrPatch(
        id="megatron.moe.grouped-gemm.available-flag",
        rebind_prefixes=("megatron",),
        target=f"{_UTIL}:grouped_gemm_is_available",
        # Same grouped_gemm_util envelope as the torch-ops fallback
        # (core_v0.5.0 through core_v0.16.x).
        version_gates=("megatron-core >=0.5,<0.17",),
        replace=_grouped_gemm_is_available,
        requires=("megatron.moe.grouped-gemm.torch-ops",),
        rationale=(
            "Consumers gate the grouped path on grouped_gemm_is_available(); "
            "with the reference ops installed the flag must tell the truth "
            "about the fallback instead of reporting a missing vendor package."
        ),
        strategy=(
            "Report True only while the torch-ops fallback is actually applied; "
            "the requires declaration keeps the flag from ever advertising a "
            "fallback that is not installed. Declines when the vendor package "
            "is present."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/transformer/moe/grouped_gemm_util.py",
        remove_when=(
            "Remove together with megatron.moe.grouped-gemm.torch-ops when a "
            "validated MUSA grouped_gemm is available."
        ),
    ),
    AttrPatch(
        id="megatron.moe.grouped-gemm.assert-noop",
        rebind_prefixes=("megatron",),
        target=f"{_UTIL}:assert_grouped_gemm_is_available",
        # Same grouped_gemm_util envelope as the torch-ops fallback
        # (core_v0.5.0 through core_v0.16.x).
        version_gates=("megatron-core >=0.5,<0.17",),
        replace=_assert_grouped_gemm_is_available,
        requires=("megatron.moe.grouped-gemm.torch-ops",),
        rationale=(
            "GroupedMLP.__init__ asserts availability before building; with the "
            "fallback installed the assertion must consult the patched flag "
            "rather than fail on the absent vendor package."
        ),
        strategy=(
            "Re-check the module's (patched) grouped_gemm_is_available at call "
            "time and keep the upstream error message for the genuinely "
            "unavailable case; becomes a no-op only while the fallback is live."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/transformer/moe/grouped_gemm_util.py",
        remove_when=(
            "Remove together with megatron.moe.grouped-gemm.torch-ops when a "
            "validated MUSA grouped_gemm is available."
        ),
    ),
)
