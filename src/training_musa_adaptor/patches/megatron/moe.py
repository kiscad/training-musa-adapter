"""Module-local MoE adapters for the MUSA stack.

These patches replace names inside ``megatron.core.transformer.moe.moe_utils``
only. The forwarding ``torch`` namespace adapts exactly the kernels the MUSA
stack cannot run (FP64 top-k) and forwards everything else unchanged, so the
upstream routing functions, group-limited top-k and router replay keep their
semantics.

A second family bridges Megatron's TE>=2.7 fused-router symbols (which MT-TE
2.0 binds to ``None``, so every ``moe_router_fusion=True`` model dies on
"fused_topk_with_score_function is not available") to torch_kernels' fused
router: same call signatures, FP32 math and write-back casting as
TransformerEngine's kernels, reference-accurate semantics.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from ... import _compat
from ..._engine import AttrPatch
from ...backends import musa_available as _musa_live

__all__ = ["PATCHES"]


logger = logging.getLogger("training_musa_adaptor")


def _fp64_topk(torch, input, k, dim, largest, sorted):
    """Reference top-k for FP64: MUSA's MuDNN TopK has no DOUBLE kernel.

    Take the indices at the input's own precision on CPU (never demote to
    FP32: near-tied scores could reorder), move the discrete indices back to
    the device and gather from the original tensor, so gradients keep flowing
    through it -- indices are discrete and need no gradient.
    """
    if dim is None:
        dim = -1  # torch.topk's documented default
    _, cpu_indices = torch.topk(
        input.detach().cpu(), k=k, dim=dim, largest=largest, sorted=sorted
    )
    indices = cpu_indices.to(input.device)
    return torch.return_types.topk((input.gather(dim, indices), indices))


class _MoeTorchProxy:
    """``torch`` namespace for moe_utils that adapts only broken kernels."""

    def __init__(self, torch):
        self._torch = torch

    def __getattr__(self, name):
        return getattr(self._torch, name)

    def topk(self, input, k, dim=-1, largest=True, sorted=True, *, out=None):
        if out is not None:
            return self._torch.topk(
                input, k, dim=dim, largest=largest, sorted=sorted, out=out
            )
        if (
            input.dtype == self._torch.float64
            and input.device.type == "musa"
            and _musa_live()
        ):
            logger.debug(
                "MoE topk FP64 reference path: shape=%s k=%s dim=%s",
                tuple(input.shape),
                k,
                dim,
            )
            return _fp64_topk(self._torch, input, k, dim, largest, sorted)
        return self._torch.topk(input, k, dim=dim, largest=largest, sorted=sorted)


def _fp64_topk_works_on_musa() -> bool:
    """Capability probe: does this torch_musa build serve fp64 topk?

    The patch exists because MuDNN topk rejects float64; a torch_musa build
    that runs the probe no longer needs the CPU-reference detour. Probed at each patch application (megatron is imported by then, so the
    device layer is active).
    """
    import torch

    try:
        values, indices = torch.topk(
            torch.arange(8, device="musa", dtype=torch.float64), 2
        )
        if values.cpu().tolist() != [7.0, 6.0] or indices.cpu().tolist() != [7, 6]:
            return False
    except Exception as exc:  # noqa: BLE001 - any failure means still broken
        logger.info(
            "moe topk fp64 probe failed, keeping the reference path (%s: %s)",
            type(exc).__name__,
            exc,
        )
        return False
    logger.info(
        "moe topk fp64-reference declined: fp64 topk works on this "
        "torch_musa build (%s)",
        _compat.distribution_version("torch-musa"),
    )
    return True


def _moe_torch_namespace(original: Any) -> Any:
    if _musa_live() and _fp64_topk_works_on_musa():
        return None
    return _MoeTorchProxy(original)


# Dtypes the MT-TE moe permutation kernel accepts. The kernel's own error text
# says "Invalid type for 16 bit", but the measured constraint is the opposite:
# float32 and float64 fail while float16/bfloat16 pass (muDNN
# ``permutation_mask.mu`` rejects the 32/64-bit key). FP8 subclasses are
# untested and deliberately left on the fused path.
def _fused_permute_unsupported(tensor) -> bool:
    """True when the fused permute kernel cannot serve this tensor's dtype."""
    import torch

    if not _musa_live() or tensor.device.type != "musa":
        return False
    return tensor.dtype in (torch.float32, torch.float64)


def _moe_permute_unfused(original: Any) -> Any:
    """Demote the fused MoE permute to upstream's reference implementation.

    ``fused_permute`` (TE ``moe_permute``) aborts in ``nvte_permute_mask`` for
    float32/float64 tokens. Upstream's ``fused=False`` branch is the reference
    implementation with identical semantics, and the paired unpermute patch
    keeps both ends on the same index format.
    """

    @functools.wraps(original)
    def permute(
        tokens,
        routing_map,
        probs=None,
        num_out_tokens=None,
        fused=False,
        drop_and_pad=False,
    ):
        demote = fused and _fused_permute_unsupported(tokens)
        if demote:
            logger.debug(
                "MoE permute unfused fallback: dtype=%s shape=%s",
                tokens.dtype,
                tuple(tokens.shape),
            )
        return original(
            tokens,
            routing_map,
            probs=probs,
            num_out_tokens=num_out_tokens,
            fused=False if demote else fused,
            drop_and_pad=drop_and_pad,
        )

    return permute


def _moe_unpermute_unfused(original: Any) -> Any:
    """Demote the fused MoE unpermute; pairs with the permute fallback."""

    @functools.wraps(original)
    def unpermute(
        permuted_tokens,
        sorted_indices,
        restore_shape,
        probs=None,
        routing_map=None,
        fused=False,
        drop_and_pad=False,
    ):
        demote = fused and _fused_permute_unsupported(permuted_tokens)
        if demote:
            logger.debug(
                "MoE unpermute unfused fallback: dtype=%s", permuted_tokens.dtype
            )
        return original(
            permuted_tokens,
            sorted_indices,
            restore_shape,
            probs=probs,
            routing_map=routing_map,
            fused=False if demote else fused,
            drop_and_pad=drop_and_pad,
        )

    return unpermute


def _moe_permute_unfused_core17(original: Any) -> Any:
    """core>=0.17 ``permute`` variant: the a2a dispatcher passes explicit
    ``tokens_per_expert``/``align_size`` keywords (added in core_v0.17.0),
    so this variant forwards ``*args``/``**kwargs`` untouched and flips only
    the ``fused`` keyword -- the form every Megatron call site uses."""

    @functools.wraps(original)
    def permute(tokens, routing_map, *args, **kwargs):
        if kwargs.get("fused") and _fused_permute_unsupported(tokens):
            logger.debug(
                "MoE permute unfused fallback (core>=0.17): dtype=%s shape=%s",
                tokens.dtype,
                tuple(tokens.shape),
            )
            return original(tokens, routing_map, *args, **{**kwargs, "fused": False})
        return original(tokens, routing_map, *args, **kwargs)

    return permute


def _moe_unpermute_unfused_core17(original: Any) -> Any:
    """core>=0.17 ``unpermute`` variant; pairs with the core17 permute
    fallback so both ends keep the same index format."""

    @functools.wraps(original)
    def unpermute(permuted_tokens, *args, **kwargs):
        if kwargs.get("fused") and _fused_permute_unsupported(permuted_tokens):
            logger.debug(
                "MoE unpermute unfused fallback (core>=0.17): dtype=%s",
                permuted_tokens.dtype,
            )
            return original(permuted_tokens, *args, **{**kwargs, "fused": False})
        return original(permuted_tokens, *args, **kwargs)

    return unpermute


def _router_fp64_linear(original):
    @functools.wraps(original)
    def linear(inp, weight, bias, router_dtype):
        import torch

        if (
            inp.device.type != "musa"
            or router_dtype != torch.float64
            or not _musa_live()
        ):
            return original(inp, weight, bias, router_dtype)
        # Keep true FP64 arithmetic and gradients. MuDNN's addmm/mm does
        # not support this router dtype; demoting would change routing ties.
        return torch.nn.functional.linear(
            inp.to(device="cpu", dtype=torch.float64),
            weight.to(device="cpu", dtype=torch.float64),
            bias.to(device="cpu", dtype=torch.float64) if bias is not None else None,
        ).to(inp.device)

    return linear


def _torch_kernels_router_entry_points():
    """torch_kernels' three fused-router entry points, or None when unusable.

    Imported lazily at patch time. A missing package declines the bridge;
    failures inside an installed package propagate with their original cause.
    """
    try:
        from torch_kernels.router import (
            fused_compute_score_for_moe_aux_loss,
            fused_moe_aux_loss,
            fused_topk_with_score_function,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "torch_kernels":
            raise
        logger.info(
            "torch_kernels fused router unavailable (%s: %s); Megatron's "
            "moe_router_fusion keeps upstream's TE>=2.7 requirement error",
            type(exc).__name__,
            exc,
        )
        return None
    return (
        fused_topk_with_score_function,
        fused_moe_aux_loss,
        fused_compute_score_for_moe_aux_loss,
    )


def _fused_router_bridge(entry_index: int):
    """Build the replace() for one fused-router symbol.

    Only ever replaces the ``None`` placeholder MT-TE 2.0 leaves behind: a
    real TransformerEngine implementation (TE>=2.7 stacks) wins, the
    DISABLE list can restore upstream, and a
    missing torch_kernels declines so Megatron's own error surfaces.
    """

    def replace(original: Any) -> Any:
        if original is not None:
            return None
        if not _musa_live():
            return None
        entry_points = _torch_kernels_router_entry_points()
        if entry_points is None:
            return None
        return entry_points[entry_index]

    return replace


_TOPK_BRIDGE = _fused_router_bridge(0)
_AUX_LOSS_BRIDGE = _fused_router_bridge(1)
_SCORE_BRIDGE = _fused_router_bridge(2)


PATCHES = (
    AttrPatch(
        id="megatron.moe.router-gating.fp64-host",
        rebind_prefixes=("megatron",),
        target="megatron.core.transformer.moe.moe_utils:router_gating_linear",
        # RouterGatingLinearFunction exists from core_v0.13.0 with the same
        # (inp, weight, bias, router_dtype) signature through core_v0.19.0,
        # the newest release line in the checkout.
        version_gates=("megatron-core >=0.13,<0.20",),
        replace=_router_fp64_linear,
        rationale="FP64 router addmm fails in MuDNN RunWithBiasAdd once delayed-wgrad construction succeeds.",
        strategy=(
            "Compute only FP64 MUSA router projection on CPU with differentiable transfers; "
            "retain FP64 logits, input/parameter gradient dtypes and bias. This costs "
            "host transfers and CPU GEMM; other precisions keep the upstream implementation."
        ),
        upstream="Megatron-LM megatron/core/transformer/moe/moe_utils.py:RouterGatingLinearFunction",
        remove_when="Remove when native FP64 router projection and gradients pass on MUSA with bias.",
    ),
    AttrPatch(
        id="megatron.moe.topk.fp64-reference",
        rebind_prefixes=("megatron",),
        target="megatron.core.transformer.moe.moe_utils:torch",
        # The fp64-router scenario this proxy protects (RouterGatingLinearFunction
        # with a free-form moe_router_dtype) exists from core_v0.13.0;
        # moe_router_dtype keeps its 'fp64' choice through core_v0.19.0, the
        # newest release line in the checkout.
        version_gates=("megatron-core >=0.13,<0.20",),
        replace=_moe_torch_namespace,
        rationale=(
            "MoE routing calls torch.topk on router scores. With "
            "moe_router_dtype='fp64' the scores are float64 and MuDNN's TopK "
            "kernel rejects that type ('TopkOut MUDNN failed in: Run'; muDNN "
            "logs 'Unsupported in data type: DOUBLE'), so every routing step of "
            "the fp64 discrepancy/aux-loss tests fails. The device-side "
            "alternatives (sort, argsort, max-with-indices) fail for float64 "
            "too, so the fallback needs a host-side index pass. Verified on "
            "torch_musa 2.7.1 / muDNN v3107 with Megatron core_v0.16.1."
        ),
        strategy=(
            "Bind moe_utils's module-local torch global to a forwarding proxy "
            "that adapts only topk: for float64 MUSA inputs it takes indices at "
            "the same precision on CPU, moves the discrete indices back and "
            "gathers values from the original tensor so gradients still flow "
            "through it. dim/largest/sorted and the (values, indices) contract "
            "are preserved, out= is delegated, and every other torch attribute "
            "forwards untouched. Non-float64 inputs and non-MUSA runtimes use "
            "the native kernel. "
            "Declines when an fp64-topk capability probe passes "
            "on this torch_musa build; this small probe is not a full kernel test."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/core/transformer/moe/moe_utils.py "
            "(_compute_topk, group_limited_topk, capacity top-k)"
        ),
        remove_when=(
            "Remove when MuDNN TopK supports float64 (or the router stops "
            "producing fp64 scores on MUSA): disable this patch id and re-run "
            "transformer/moe/test_moe_layer_discrepancy.py, test_aux_loss.py and "
            "the fp64 routing cases; delete only if the native path passes."
        ),
    ),
    AttrPatch(
        id="megatron.moe.permutation.unfused-musa",
        rebind_prefixes=("megatron",),
        target="megatron.core.transformer.moe.moe_utils:permute",
        # The fused permute branch (fused_permute, drop_and_pad) that this
        # patch demotes exists from core_v0.11.0. Capped at <0.17:
        # core_v0.17.0 adds tokens_per_expert/align_size and its a2a
        # dispatcher passes them, so that contract is carried by the separate
        # core17 variant below.
        version_gates=("megatron-core >=0.11,<0.17",),
        replace=_moe_permute_unfused,
        rationale=(
            "The token dispatcher's fused permute calls TE's moe_permute, whose "
            "MUSA kernel (nvte_permute_mask) aborts for float32 tokens with "
            "'Invalid type for 16 bit' despite the 16-bit wording: measured on "
            "torch_musa 2.7.1, float32/float64 fail while float16/bfloat16 pass. "
            "This breaks every MoE a2a-token-dispatcher case that runs with "
            "moe_permute_fusion enabled and non-16-bit activations."
        ),
        strategy=(
            "Demote exactly the fused branch to upstream's own reference "
            "implementation by re-calling the original with fused=False when the "
            "token dtype is a confirmed-broken one on a live MUSA device. "
            "probs/num_out_tokens/drop_and_pad and the (permuted tokens, "
            "permuted probs, sorted indices) contract are unchanged, gradients "
            "flow through upstream's scatter path, and float16/bfloat16/FP8 "
            "inputs keep the fused kernel. Use together with the unpermute patch "
            "so both ends use the same index format."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/transformer/moe/moe_utils.py:permute",
        remove_when=(
            "Remove when the MUSA TE permutation kernel supports 32-bit inputs: "
            "disable both permutation patch ids and re-run "
            "transformer/moe/test_a2a_token_dispatcher.py forward/backward; "
            "delete only if the fused path passes."
        ),
    ),
    AttrPatch(
        id="megatron.moe.unpermutation.unfused-musa",
        rebind_prefixes=("megatron",),
        target="megatron.core.transformer.moe.moe_utils:unpermute",
        # Same fused-permute envelope as the permute companion (unpermute
        # gains pad_offsets in core_v0.17.0 -- carried by the core17 variant
        # below).
        version_gates=("megatron-core >=0.11,<0.17",),
        replace=_moe_unpermute_unfused,
        requires=("megatron.moe.permutation.unfused-musa",),
        rationale=(
            "The fused unpermute uses the same MUSA permutation kernel family as "
            "the fused permute. Restoring tokens must stay on the same index "
            "format as the permute that produced them: mixing a non-fused "
            "permute with the fused unpermute (or vice versa) silently "
            "scrambles token order."
        ),
        strategy=(
            "Demote the fused branch to upstream's reference implementation under "
            "the same dtype condition as the permute patch, keeping probs, "
            "routing_map, restore_shape and drop_and_pad semantics. Declared as "
            "requiring the permute companion so ONLY/DISABLE cannot activate "
            "unpermute without permute; selecting permute alone is not a complete "
            "dispatch/restore compatibility path."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/transformer/moe/moe_utils.py:unpermute",
        remove_when=(
            "Remove together with megatron.moe.permutation.unfused-musa after the "
            "MUSA TE permutation kernel supports 32-bit inputs and the a2a "
            "dispatcher forward/backward passes on the fused path."
        ),
    ),
    AttrPatch(
        id="megatron.moe.permutation.unfused-musa.core17",
        rebind_prefixes=("megatron",),
        target="megatron.core.transformer.moe.moe_utils:permute",
        # core_v0.17.0 adds tokens_per_expert/align_size to permute and its
        # a2a dispatcher passes them; a separate variant keeps the 0.11-0.16
        # patch simple. Verified through core_v0.19.0, the newest release
        # line in the checkout.
        version_gates=("megatron-core >=0.17,<0.20",),
        replace=_moe_permute_unfused_core17,
        rationale=(
            "Same MUSA fused-permute abort as the 0.11-0.16 patch, on the "
            "core_v0.17.0+ contract: permute gains tokens_per_expert/align_size "
            "and the a2a dispatcher passes them explicitly, so the earlier "
            "fixed-signature wrapper would TypeError. The MUSA TE permutation "
            "kernel still rejects float32/float64 tokens on this stack."
        ),
        strategy=(
            "Separate variant patch (not a wider signature bolted onto the "
            "validated 0.11-0.16 one): forward *args/**kwargs untouched and "
            "flip only the fused keyword -- the form every Megatron call site "
            "uses -- when the token dtype is a confirmed-broken one on a live "
            "MUSA device. Positional fused callers are not adapted and keep "
            "upstream behavior. Use together with the core17 unpermute variant."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/transformer/moe/moe_utils.py:permute",
        remove_when=(
            "Remove when the MUSA TE permutation kernel supports 32-bit inputs: "
            "disable both core17 permutation patch ids and re-run "
            "transformer/moe/test_a2a_token_dispatcher.py forward/backward; "
            "delete only if the fused path passes."
        ),
    ),
    AttrPatch(
        id="megatron.moe.unpermutation.unfused-musa.core17",
        rebind_prefixes=("megatron",),
        target="megatron.core.transformer.moe.moe_utils:unpermute",
        # Same core17 contract as the permute variant (unpermute gains
        # pad_offsets in core_v0.17.0). Verified through core_v0.19.0.
        version_gates=("megatron-core >=0.17,<0.20",),
        replace=_moe_unpermute_unfused_core17,
        requires=("megatron.moe.permutation.unfused-musa.core17",),
        rationale=(
            "The fused unpermute uses the same MUSA permutation kernel family "
            "as the fused permute, and core_v0.17.0 adds pad_offsets to its "
            "signature. Restoring tokens must stay on the same index format as "
            "the permute that produced them: mixing a non-fused permute with "
            "the fused unpermute (or vice versa) silently scrambles token order."
        ),
        strategy=(
            "Separate variant patch paired with the core17 permute fallback: "
            "forward *args/**kwargs untouched and flip only the fused keyword "
            "under the same dtype condition. Declared as requiring the core17 "
            "permute variant so ONLY/DISABLE cannot activate unpermute without "
            "permute."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/transformer/moe/moe_utils.py:unpermute",
        remove_when=(
            "Remove together with megatron.moe.permutation.unfused-musa.core17 "
            "after the MUSA TE permutation kernel supports 32-bit inputs and "
            "the a2a dispatcher forward/backward passes on the fused path."
        ),
    ),
    AttrPatch(
        id="megatron.moe.fused-router.topk-with-score-function",
        rebind_prefixes=("megatron",),
        target="megatron.core.extensions.transformer_engine:fused_topk_with_score_function",
        # The TE>=2.7 fused-router import guards (and the moe_router_fusion
        # config that consumes them) exist from core_v0.14.0; the same
        # is_te_min_version('2.7.0.dev') guard is still in place at
        # core_v0.19.0, the newest release line in the checkout.
        version_gates=("megatron-core >=0.14,<0.20",),
        replace=_TOPK_BRIDGE,
        rationale=(
            "Megatron's moe_router_fusion path calls TE>=2.7's "
            "fused_topk_with_score_function for every routing step "
            "(moe_utils.topk_softmax_with_capacity, fused=True). MT-TE 2.0 "
            "binds the symbol to None under its is_te_min_version('2.7.0.dev') "
            "import guard, so 12 megatron-FSDP cases and any "
            "moe_router_fusion=True training die on 'fused_topk_with_score_"
            "function is not available. Please install TE >= 2.6.0'."
        ),
        strategy=(
            "Bridge the None placeholder to torch_kernels' fused router "
            "(torch_kernels.router.fused_topk_with_score_function): identical "
            "keyword signature, TransformerEngine's precision contract (FP32 "
            "math, write-back cast, sigmoid expert_bias steering selection "
            "only, post-softmax over the selected scores, group-limited "
            "routing, 1e-20 sigmoid epsilon) and differentiable probabilities "
            "through the reference composition; torch_kernels' own tests pin "
            "it bit-exactly against Megatron's reference math. Only replaces "
            "the None placeholder -- a real TE>=2.7 implementation wins; "
            "disabling these router patches and a missing "
            "torch_kernels both decline to keep upstream's error."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/core/transformer/moe/moe_utils.py:"
            "topk_softmax_with_capacity; TransformerEngine pytorch/router.py"
        ),
        remove_when=(
            "Remove when MT-TransformerEngine ships the fused router: disable "
            "this id and re-run the moe_router_fusion cases (megatron-FSDP "
            "test_mcore_fully_sharded_data_parallel, transformer/moe router "
            "tests); delete only if the native symbols pass."
        ),
    ),
    AttrPatch(
        id="megatron.moe.fused-router.moe-aux-loss",
        rebind_prefixes=("megatron",),
        target="megatron.core.extensions.transformer_engine:fused_moe_aux_loss",
        # Same TE>=2.7 fused-router guard envelope as the topk bridge
        # (core_v0.14.0; verified through core_v0.19.0).
        version_gates=("megatron-core >=0.14,<0.20",),
        replace=_AUX_LOSS_BRIDGE,
        rationale=(
            "The same TE>=2.7 import guard leaves fused_moe_aux_loss as None; "
            "moe_utils.switch_load_balancing_loss_func(fused=True) raises for "
            "every moe_router_fusion model with a nonzero moe_aux_loss_coeff."
        ),
        strategy=(
            "Bridge the None placeholder to torch_kernels' "
            "fused_moe_aux_loss (same keyword signature, FP32 accumulation, "
            "probs-dtype write-back, int or 0-dim-tensor total_num_tokens for "
            "graph-safe replays, differentiable). Same decline rules as the "
            "topk bridge."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/core/transformer/moe/moe_utils.py:"
            "switch_load_balancing_loss_func; TransformerEngine "
            "pytorch/router.py:fused_moe_aux_loss"
        ),
        remove_when=(
            "Remove together with megatron.moe.fused-router."
            "topk-with-score-function after MT-TransformerEngine ships the "
            "fused router."
        ),
    ),
    AttrPatch(
        id="megatron.moe.fused-router.score-for-aux-loss",
        rebind_prefixes=("megatron",),
        target="megatron.core.extensions.transformer_engine:"
        "fused_compute_score_for_moe_aux_loss",
        # Same TE>=2.7 fused-router guard envelope as the topk bridge
        # (core_v0.14.0; verified through core_v0.19.0).
        version_gates=("megatron-core >=0.14,<0.20",),
        replace=_SCORE_BRIDGE,
        rationale=(
            "The same TE>=2.7 import guard leaves "
            "fused_compute_score_for_moe_aux_loss as None; "
            "moe_utils.compute_routing_scores_for_aux_loss(fused=True) raises "
            "for sequence-aux-loss routers under moe_router_fusion."
        ),
        strategy=(
            "Bridge the None placeholder to torch_kernels' "
            "fused_compute_score_for_moe_aux_loss (routing map + FP32 "
            "normalized scores, differentiable scores). Same decline rules as "
            "the topk bridge."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/core/transformer/moe/moe_utils.py:"
            "compute_routing_scores_for_aux_loss; TransformerEngine "
            "pytorch/router.py:fused_compute_score_for_moe_aux_loss"
        ),
        remove_when=(
            "Remove together with megatron.moe.fused-router."
            "topk-with-score-function after MT-TransformerEngine ships the "
            "fused router."
        ),
    ),
)
