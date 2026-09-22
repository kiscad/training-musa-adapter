"""Fused RoPE on MUSA: apex kernels behind Megatron's Transformer Engine seam.

Megatron takes its fused rotary kernels from Transformer Engine.  When that
import does not produce them, ``megatron.core.models.common.embeddings.rope_utils``
falls back to ``None`` and ``TransformerConfig`` rejects ``apply_rope_fusion``
before the first step -- and ``apply_rope_fusion`` is argparse's *default*
(``--no-rope-fusion`` is the opt-out), so the rejection aborts runs that never
asked for anything unusual.

On MUSA the TE kernels are missing for a structural reason: the version-gated
import in ``megatron/core/extensions/transformer_engine.py`` selects
``transformer_engine.pytorch.attention.rope`` (the TE >= 2.3.0 location) whenever
Megatron believes TE is new enough, and the MUSA TE development tree has no such
submodule -- its ``apply_rotary_pos_emb`` still lives in
``transformer_engine.pytorch.attention``.  The ``ImportError`` is swallowed by
upstream's own ``except ImportError: pass``.

Two kernel sources fill that seam, in priority order:

1. ``aten::rope`` -- torch's fused RoPE op (top-level ``torch.rope`` /
   ``torch.ops.aten.rope``).  The torch_musa build ships the MUSA kernel
   (``at::musa::RopeOut`` plus ``_fused_rope_forward``/``_fused_rope_backward``
   for autograd); CPU/CUDA have no kernel and raise.  Verified on MUSA
   (8x MTT S5000, torch_musa 2.7.1+1569808, muDNN v3107): fp32 matches the
   unfused reference to 4.8e-07 for BOTH interleaved modes, bf16 stays
   within one bf16 ULP of the input magnitude (the kernel computes in fp32
   internally -- single rounding, more accurate than the stepwise bf16
   reference), and gradients match (fp32 bit-identical).  Performance
   (MTT S5000, bf16, S=4096/8192, B=1-2, H=8-32, D=128): 3.4-4.9x faster
   than apex, 7.8-11.1x faster than the unfused reference.  The kernel does
   NOT support head dims wider than ``freq_cis`` -- the wrapper splits the
   passthrough tail like megatron's reference.  thd/varlen is not
   integrated (no cu_seqlens form; apex keeps that path).
2. Moore Threads' apex fork fused kernels -- the fallback when the torch op
   is unavailable.

Neither binding ever overrides a kernel Transformer Engine actually provided.
"""

from __future__ import annotations

import functools
import logging
from copy import copy
from typing import Any

from ..._engine import AttrPatch

__all__ = ["PATCHES"]

logger = logging.getLogger("training_musa_adaptor")

#: The one module all three patches share: Megatron's rope dispatch and its two
#: optional fused kernels.
_ROPE_UTILS = "megatron.core.models.common.embeddings.rope_utils"

#: One warning per reason: the fused path runs on every attention layer.
_warned: set[str] = set()

#: Marks the wrappers this module installs, so the dispatcher can tell an
#: apex kernel from one Transformer Engine provided (same convention as the
#: LayerNorm fallback's ``_tma_fallback``).
_MARKER = "_tma_apex_rope"


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message, *args)


def _bound_kernel(attr: str) -> Any:
    """What ``rope_utils`` currently binds to ``attr``, without importing it.

    Called while that module is being patched, so it is already in
    ``sys.modules``; a direct import would add a Megatron import to a code path
    that unit tests exercise with stubs.
    """
    import sys

    module = sys.modules.get(_ROPE_UTILS)
    return getattr(module, attr, None) if module is not None else None


def _is_apex_kernel(candidate: Any) -> bool:
    return bool(getattr(candidate, _MARKER, False))


def _kernel_provider(kernel: Any) -> str | None:
    """Which provider a bound fused-RoPE kernel comes from."""
    if kernel is None:
        return None
    if getattr(kernel, _MARKER, False):
        return "apex"
    if getattr(kernel, "_tma_aten_rope", False):
        return "aten"
    return None


def _aten_rope_op() -> Any | None:
    """The torch fused RoPE op (``aten::rope``), or ``None`` when unusable.

    Stock torch 2.7+ declares the op (``torch.rope`` top-level alias and
    ``torch.ops.aten.rope``); only the torch_musa build ships a compute
    kernel -- CPU/CUDA calls raise "rope only supported in torch_musa" --
    so the probe requires the MUSA stack to be present.  Capability notes:
    ``rotary_interleaved``/``multi_latent_attention`` flags exist in the
    schema; interleaved was verified against megatron's adjacent-pair freqs
    layout on MUSA (fp32, 4.8e-07 vs the unfused reference), mla stays
    unused (megatron 0.16 does not route it here).
    """
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch-free import chains
        return None
    op = getattr(torch, "rope", None)
    if op is None:
        op = getattr(getattr(torch.ops, "aten", None), "rope", None)
    if op is None:
        return None
    try:
        import importlib.util

        if importlib.util.find_spec("torch_musa") is None:
            return None
    except Exception:  # noqa: BLE001 - a broken lookup is an undecided probe
        return None
    return op


def _apex_kernels() -> tuple[Any, Any] | None:
    """apex's fused RoPE pair, or ``None`` when this stack has no usable one.

    torch and apex are imported here, never at module scope: ``patches/`` is
    imported inside every process that imports the patch set, and only a process
    that reaches ``rope_utils`` needs these kernels.

    The compiled ``fused_rotary_positional_embedding`` extension is probed too.
    "apex imports" does not prove "the kernel runs" on MUSA (the same trap the
    fused LayerNorm fallback documents), and a missing extension must degrade to
    upstream's startup-time refusal rather than a crash in the first attention
    layer.
    """
    try:
        import fused_rotary_positional_embedding  # noqa: F401  (probe only)
        from apex.transformer.functional import (
            fused_apply_rotary_pos_emb,
            fused_apply_rotary_pos_emb_thd,
        )
    except ModuleNotFoundError as exc:
        if exc.name not in ("apex", "fused_rotary_positional_embedding"):
            raise
        _warn_once(
            "apex-missing",
            "training-musa-adaptor: apex's fused RoPE is unavailable (%s: %s); "
            "apply_rope_fusion stays unavailable, so disable it with "
            "--no-rope-fusion.",
            type(exc).__name__,
            exc,
        )
        return None
    return fused_apply_rotary_pos_emb, fused_apply_rotary_pos_emb_thd


def _apex_fused_bshd(original: Any) -> Any:
    """Upstream-shaped ``sbhd`` fused RoPE backed by apex's kernel."""
    if original is not None:
        return None
    kernels = _apex_kernels()
    if kernels is None:
        return None
    apex_fused, _ = kernels

    @functools.wraps(apex_fused)
    def fused_apply_rotary_pos_emb(
        t, freqs, transpose_output_memory: bool = False, interleaved: bool = False
    ):
        """Apply rotary positional embedding to ``t`` in ``sbhd`` format.

        ``transpose_output_memory`` is native to apex and is honoured, unlike the
        current TE wrapper which warns and ignores it.  Apex has no interleaved
        variant: ``apply_rotary_pos_emb`` sends those models to the unfused
        kernel, so this branch is only reachable by a direct caller.
        """
        if interleaved:
            raise NotImplementedError(
                "apex's fused RoPE has no interleaved variant on MUSA; use "
                "apply_rotary_pos_emb (which falls back to the unfused kernel "
                "for rotary_interleaved configs) or disable --rope-fusion"
            )
        return apex_fused(t, freqs, transpose_output_memory)

    setattr(fused_apply_rotary_pos_emb, _MARKER, True)
    return fused_apply_rotary_pos_emb


def _aten_fused_bshd(original: Any) -> Any:
    """Upstream-shaped ``sbhd`` fused RoPE backed by ``aten::rope``.

    Preferred over the apex binding when the op is available.  Verified on
    MUSA against megatron's unfused reference for both interleaved modes
    (fp32 4.8e-07; bf16 within one input-magnitude bf16 ULP -- the kernel
    computes in fp32 internally, single rounding).  The MUSA kernel rejects
    head dims wider than ``freq_cis``, so the wrapper splits the passthrough
    tail exactly like megatron's reference.  The op has no
    ``transpose_output_memory`` argument (accepted and ignored -- the same
    contract TE documents), no context-parallel form (thd stays on apex),
    and no mscale (the upstream fused path does not apply it either).
    """
    module = getattr(original, "__module__", "") or ""
    if original is not None and module != "megatron.core.extensions.transformer_engine":
        return None  # Preserve native TE and third-party implementations.
    if getattr(original, _MARKER, False) or getattr(original, "_tma_aten_rope", False):
        return None  # never double-wrap this module's own bindings
    op = _aten_rope_op()
    if op is None:
        return None

    def fused_apply_rotary_pos_emb(
        t, freqs, transpose_output_memory: bool = False, interleaved: bool = False
    ):
        """Apply rotary positional embedding to ``t`` in ``sbhd`` format.

        ``freqs`` ([S, 1, 1, D] or [S, D], real) is flattened to [S, D] --
        positions are shared across the batch, and the op broadcasts the row
        over batch and heads (the verified production call shape).  Head dims
        wider than ``freq_cis`` pass through unrotated, split exactly like
        megatron's reference.
        """
        if getattr(getattr(t, "device", None), "type", None) != "musa":
            raise NotImplementedError(
                "aten::rope only has a torch_musa kernel; the dispatcher "
                "routes non-MUSA inputs to the unfused kernel"
            )
        rot_dim = freqs.shape[-1]
        t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
        out = op(
            t_rot,
            freqs.reshape(-1, rot_dim),
            rotary_interleaved=interleaved,
            batch_first=False,
        )
        if t_pass.numel():
            import torch

            out = torch.cat((out, t_pass), dim=-1)
        return out

    fused_apply_rotary_pos_emb._tma_aten_rope = True
    return fused_apply_rotary_pos_emb


def _apex_fused_thd(original: Any) -> Any:
    """Upstream-shaped ``thd`` fused RoPE backed by apex's kernel."""
    if original is not None:
        return None
    kernels = _apex_kernels()
    if kernels is None:
        return None
    _, apex_fused_thd = kernels

    @functools.wraps(apex_fused_thd)
    def fused_apply_rotary_pos_emb_thd(
        t, cu_seqlens, freqs, cp_size: int = 1, cp_rank: int = 0
    ):
        """Apply rotary positional embedding to ``t`` in ``thd`` format.

        Like the Transformer Engine kernel Megatron calls on NVIDIA, apex expects
        the padded layout (``cu_seqlens`` = ``cu_seqlens_*_padded``); Megatron
        passes exactly that when the packed sequence is padded.  There is no
        context-parallel variant, so ``cp_size > 1`` is refused here and routed
        to the unfused path by ``apply_rotary_pos_emb``.
        """
        if cp_size != 1:
            raise NotImplementedError(
                "apex's fused RoPE has no context-parallel variant on MUSA "
                f"(cp_size={cp_size}); use apply_rotary_pos_emb (which falls "
                "back to the unfused kernel for context parallel) or disable "
                "--rope-fusion"
            )
        return apex_fused_thd(t, cu_seqlens, freqs)

    setattr(fused_apply_rotary_pos_emb_thd, _MARKER, True)
    return fused_apply_rotary_pos_emb_thd


def _apex_fused_thd_core17(original: Any) -> Any:
    """core>=0.17 ``thd`` variant: the dispatcher passes ``interleaved=``
    (added in core_v0.17.0). Apex has no interleaved thd kernel and the
    dispatcher patch demotes those configs first, so the keyword exists to
    keep the call contract and is refused if it is ever True."""
    if original is not None:
        return None
    kernels = _apex_kernels()
    if kernels is None:
        return None
    _, apex_fused_thd = kernels

    @functools.wraps(apex_fused_thd)
    def fused_apply_rotary_pos_emb_thd(
        t,
        cu_seqlens,
        freqs,
        cp_size: int = 1,
        cp_rank: int = 0,
        interleaved: bool = False,
    ):
        """Apply rotary positional embedding to ``t`` in ``thd`` format.

        Same padded-layout contract as the 0.13-0.16 binding; the added
        ``interleaved`` keyword is accepted for the core_v0.17.0+ call shape
        and refused, exactly like the sbhd kernel's interleaved handling.
        """
        if interleaved:
            raise NotImplementedError(
                "apex's fused RoPE has no interleaved variant on MUSA; use "
                "apply_rotary_pos_emb (which falls back to the unfused kernel "
                "for rotary_interleaved configs) or disable --rope-fusion"
            )
        if cp_size != 1:
            raise NotImplementedError(
                "apex's fused RoPE has no context-parallel variant on MUSA "
                f"(cp_size={cp_size}); use apply_rotary_pos_emb (which falls "
                "back to the unfused kernel for context parallel) or disable "
                "--rope-fusion"
            )
        return apex_fused_thd(t, cu_seqlens, freqs)

    setattr(fused_apply_rotary_pos_emb_thd, _MARKER, True)
    return fused_apply_rotary_pos_emb_thd


def _context_parallel_group() -> Any:
    """Upstream's own fallback for a caller that passes no ``cp_group``."""
    from megatron.core import parallel_state

    return parallel_state.get_context_parallel_group()


def _unfusable_reason(
    config: Any,
    cu_seqlens: Any,
    cp_group: Any,
    *,
    bshd_provider: str | None,
    thd_provider: str | None,
    input_device_type: str | None,
) -> str:
    """Why the bound kernels cannot serve this call, or ``""`` when they can.

    Only this module's own bindings (apex/aten) are demoted here: a kernel
    Transformer Engine provided is not demoted, whatever it supports.
    """
    if cu_seqlens is None:
        if bshd_provider == "aten":
            # The op ships only a torch_musa kernel; interleaved was verified
            # against megatron's freqs layout on MUSA (fp32, 4.8e-07).
            if input_device_type != "musa":
                return "non-musa"
            return ""
        if bshd_provider == "apex" and getattr(config, "rotary_interleaved", False):
            return "interleaved"
        return ""
    if thd_provider != "apex":
        return ""
    if getattr(config, "rotary_interleaved", False):
        return "interleaved"
    if cp_group is None:
        cp_group = _context_parallel_group()
    return "context-parallel" if cp_group is not None and cp_group.size() > 1 else ""


def _unfused_where_apex_cannot_fuse(original: Any) -> Any:
    """Send the combinations apex cannot fuse to upstream's unfused branch.

    Two configurations reach the fused call without a kernel that can serve
    them: ``rotary_interleaved`` models (apex has no interleaved variant) and
    packed sequences under context parallelism (apex has no CP variant).  Rather
    than aborting a validated configuration in the middle of a step, the request
    is demoted to the unfused implementation Megatron already ships -- which is
    what this patch set recommends as the conservative MUSA path anyway.

    Inspect the actual kernels at call time, so selection/reload and registration
    order cannot leave stale assumptions. A shallow config copy selects upstream's
    unfused branch without mutating the shared model configuration. Native kernels
    and calls without fusion pass through unchanged.
    """

    @functools.wraps(original)
    def apply_rotary_pos_emb(
        t, freqs, config, cu_seqlens=None, mscale: float = 1.0, cp_group=None
    ):
        reason = ""
        if config.apply_rope_fusion:
            reason = _unfusable_reason(
                config,
                cu_seqlens,
                cp_group,
                bshd_provider=_kernel_provider(
                    _bound_kernel("fused_apply_rotary_pos_emb")
                ),
                thd_provider=_kernel_provider(
                    _bound_kernel("fused_apply_rotary_pos_emb_thd")
                ),
                input_device_type=getattr(getattr(t, "device", None), "type", None),
            )
        if (
            not reason
            and config.apply_rope_fusion
            and getattr(config, "rotary_interleaved", False)
        ):
            # Core can expose a native TE wrapper even when TE < 2.3 rejects
            # interleaved RoPE. Respect Core's real version guard, and only
            # adapt its exact binding (never infer capabilities of a foreign one).
            import sys

            extension = sys.modules.get("megatron.core.extensions.transformer_engine")
            name = (
                "fused_apply_rotary_pos_emb"
                if cu_seqlens is None
                else "fused_apply_rotary_pos_emb_thd"
            )
            kernel = _bound_kernel(name)
            if (
                extension is not None
                and kernel is not None
                and kernel is vars(extension).get(name)
            ):
                version_check = vars(extension).get("is_te_min_version")
                if callable(version_check) and not version_check("2.3.0"):
                    reason = "interleaved"
        if not reason:
            return original(
                t,
                freqs,
                config=config,
                cu_seqlens=cu_seqlens,
                mscale=mscale,
                cp_group=cp_group,
            )
        _warn_once(
            reason,
            "training-musa-adaptor: the installed fused RoPE cannot serve %s on MUSA; "
            "using Megatron's unfused rotary embedding for those layers.",
            (
                "rotary_interleaved models"
                if reason == "interleaved"
                else (
                    "non-MUSA inputs (aten::rope has only a torch_musa kernel)"
                    if reason == "non-musa"
                    else "packed sequences with context parallel > 1"
                )
            ),
        )
        local_config = copy(config)
        local_config.apply_rope_fusion = False
        return original(
            t,
            freqs,
            config=local_config,
            cu_seqlens=cu_seqlens,
            mscale=mscale,
            cp_group=cp_group,
        )

    return apply_rotary_pos_emb


PATCHES = (
    AttrPatch(
        id="megatron.embeddings.fused-rope.aten",
        rebind_prefixes=("megatron",),
        target=f"{_ROPE_UTILS}:fused_apply_rotary_pos_emb",
        # torch 的融合 RoPE 算子（aten::rope，torch_musa 提供 MUSA 内核）优先于
        # apex 绑定；两种 interleaved 模式已在 MUSA 上与 unfused 参考对数一致
        # （rotary_interleaved 与非 MUSA 输入由 dispatcher 降级到 unfused）。
        # thd 路径维持 apex 绑定：算子没有 cu_seqlens 形态。
        version_gates=("megatron-core >=0.13,<0.20",),
        replace=_aten_fused_bshd,
        rationale=(
            "The torch stack ships a fused RoPE op (aten::rope; torch_musa "
            "implements at::musa::RopeOut plus _fused_rope_forward/_backward) "
            "that the reference Megatron-MUSA adapter calls on this exact "
            "seam. Measured on MUSA (MTT S5000, bf16, S=4096/8192): 3.4-4.9x "
            "faster than the apex kernel and 7.8-11.1x faster than megatron's "
            "unfused reference. rope_utils still binds None when the MT-TE "
            "fork leaves the TE >= 2.3 import unsatisfied, so the op fills "
            "the same seam with a vendor kernel instead of apex."
        ),
        strategy=(
            "Bind an aten::rope-backed wrapper with the upstream call shape "
            "(t sbhd, freqs flattened to [S, D], rotary_interleaved=False, "
            "batch_first=False) when the symbol is None or the Megatron TE "
            "wrapper; preserve native TE and third-party kernels. Registered "
            "before the apex binding so the torch op wins "
            "when available and apex takes over when it is not. Scope is "
            "passed through and verified on MUSA against megatron's unfused "
            "reference; thd stays on apex "
            "Disable this patch ID to leave upstream's verdict untouched."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/core/models/common/embeddings/"
            "rope_utils.py; pytorch torch/nn/functional.py rope "
            "(aten::rope); torch_musa at::musa::RopeOut kernel"
        ),
        remove_when=(
            "Remove when the MUSA Transformer Engine provides the "
            "transformer_engine.pytorch.attention.rope module Megatron's "
            "version gate looks for, or when the aten::rope path is shown "
            "inferior on MUSA; re-run bshd fused/unfused parity plus a "
            "pretrain step on MUSA before deleting."
        ),
    ),
    AttrPatch(
        id="megatron.embeddings.fused-rope.apex",
        rebind_prefixes=("megatron",),
        target=f"{_ROPE_UTILS}:fused_apply_rotary_pos_emb",
        # The TE >= 2.3 version-gated import that leaves these symbols None
        # exists from core_v0.13.0 (rope_utils.py itself from core_v0.10.0,
        # where TE is imported directly and provides kernels). The upstream
        # call shape (t, freqs, interleaved=...) is unchanged through
        # core_v0.19.0, the newest release line in the checkout.
        version_gates=("megatron-core >=0.13,<0.20",),
        replace=_apex_fused_bshd,
        rationale=(
            "Megatron imports its fused RoPE from Transformer Engine, gated on a "
            "TE >= 2.3.0 check that selects transformer_engine.pytorch.attention.rope; "
            "the MUSA TE development tree has no such submodule, its "
            "apply_rotary_pos_emb still lives in transformer_engine.pytorch.attention, "
            "and upstream swallows the ImportError. rope_utils therefore keeps None and "
            "TransformerConfig rejects apply_rope_fusion -- argparse's default -- at "
            "startup. apex's fork of the same NVIDIA kernel ships on MUSA and computes "
            "the identical sbhd rotation."
        ),
        strategy=(
            "Bind an apex-backed kernel with the upstream wrapper's signature only when "
            "Transformer Engine left the symbol None, so a future MUSA TE rope "
            "implementation still wins. transpose_output_memory is honoured natively; "
            "interleaved has no apex variant and is refused here, with "
            "apply_rotary_pos_emb routing those configs to the unfused kernel. "
            "Disable this patch ID to leave upstream's verdict untouched."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/models/common/embeddings/rope_utils.py",
        remove_when=(
            "Remove after the MUSA Transformer Engine provides the "
            "transformer_engine.pytorch.attention.rope module Megatron's version gate "
            "looks for, and bshd fused/unfused bf16 forward/backward parity plus a "
            "pretrain step are re-validated on MUSA."
        ),
    ),
    AttrPatch(
        id="megatron.embeddings.fused-rope-thd.apex",
        rebind_prefixes=("megatron",),
        target=f"{_ROPE_UTILS}:fused_apply_rotary_pos_emb_thd",
        # Same TE >= 2.3 gated import seam as the sbhd kernel; the thd symbol
        # gates Megatron's whole fused-RoPE availability check from
        # core_v0.13.0. Capped at <0.17: core_v0.17.0's dispatcher passes
        # interleaved= to the thd kernel, which the simple 0.13-0.16 wrapper
        # does not accept -- that contract is carried by the separate core17
        # variant below.
        version_gates=("megatron-core >=0.13,<0.17",),
        replace=_apex_fused_thd,
        rationale=(
            "The packed (thd) half of the same missing Transformer Engine import. "
            "Megatron also gates the whole fused-RoPE availability check on this "
            "symbol, so without it both the sbhd and the thd configurations are "
            "rejected even though apex provides kernels for both."
        ),
        strategy=(
            "Bind apex's padded-thd kernel (cu_seqlens = cu_seqlens_*_padded, exactly "
            "what Megatron passes and what the TE kernel expects) when Transformer "
            "Engine left the symbol None. cp_size=1 only: apex has no "
            "context-parallel variant, and apply_rotary_pos_emb demotes cp_size > 1 "
            "to the unfused CP-aware kernel. Disable this patch ID to leave upstream untouched."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/models/common/embeddings/rope_utils.py",
        remove_when=(
            "Remove together with the sbhd fallback, after MUSA TE supplies the fused "
            "thd kernel and padded-layout packed-sequence parity has been re-validated "
            "on MUSA."
        ),
    ),
    AttrPatch(
        id="megatron.embeddings.fused-rope-thd.apex.core17",
        rebind_prefixes=("megatron",),
        target=f"{_ROPE_UTILS}:fused_apply_rotary_pos_emb_thd",
        # core_v0.17.0's dispatcher passes interleaved= to the thd kernel;
        # this variant carries that call contract so the 0.13-0.16 binding
        # stays simple. Verified through core_v0.19.0, the newest release
        # line in the checkout.
        version_gates=("megatron-core >=0.17,<0.20",),
        replace=_apex_fused_thd_core17,
        rationale=(
            "core_v0.17.0 extends the dispatcher's thd call with interleaved= "
            "(rotary_interleaved models), which the plain 0.13-0.16 wrapper "
            "does not accept. The seam is unchanged otherwise: the TE >= 2.3 "
            "gated import still leaves the symbol None on the MUSA Transformer "
            "Engine tree, and apex's padded-thd kernel computes the identical "
            "rotation for the non-interleaved configs that reach it."
        ),
        strategy=(
            "Separate variant patch (not a wider signature bolted onto the "
            "validated 0.13-0.16 one): accept the interleaved keyword to keep "
            "the core_v0.17.0+ call contract and refuse interleaved=True "
            "exactly like the sbhd kernel -- apex has no interleaved variant, "
            "and the dispatcher patch demotes those configs to the unfused "
            "kernel first. The cp_size=1 refusal is unchanged. Declines "
            "whenever Transformer Engine provided a kernel."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/models/common/embeddings/rope_utils.py",
        remove_when=(
            "Remove together with the sbhd fallback and its core17 variant, "
            "after MUSA TE supplies the fused thd kernel and padded-layout "
            "packed-sequence parity has been re-validated on MUSA."
        ),
    ),
    AttrPatch(
        id="megatron.embeddings.rope-fusion.unfused-fallback",
        rebind_prefixes=("megatron",),
        target=f"{_ROPE_UTILS}:apply_rotary_pos_emb",
        # The dispatcher signature this wrapper mirrors (config, cu_seqlens,
        # mscale, cp_group) is complete from core_v0.13.0. core_v0.19.0 adds
        # an optional trailing mla_rotary_interleaved parameter that no
        # Megatron caller passes yet, so the wrapper stays call-compatible
        # through core_v0.19.0, the newest release line in the checkout.
        version_gates=("megatron-core >=0.13,<0.20",),
        replace=_unfused_where_apex_cannot_fuse,
        rationale=(
            "With the apex kernels installed, apply_rope_fusion becomes the default "
            "path again, but two valid configurations ask for a fused kernel that does "
            "not exist on MUSA: rotary_interleaved (apex has no interleaved variant) and "
            "packed sequences under context parallelism (apex has no CP variant). "
            "Without this seam they abort in the first attention layer of a run whose "
            "arguments already passed validation."
        ),
        strategy=(
            "Wrap the dispatcher: keep upstream's fused selection whenever apex can "
            "serve it, inspecting currently bound kernels at call time. For unsupported "
            "combinations pass a shallow config copy with apply_rope_fusion=False to "
            "upstream, warning once per reason without mutating shared config. "
            "Core native TE wrappers that reject interleaved RoPE before TE 2.3 "
            "also select the unfused route. Other native or missing kernels pass "
            "through, preserving upstream errors; "
            "the dispatcher neither imports apex nor requires kernel patch ordering."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/models/common/embeddings/rope_utils.py",
        remove_when=(
            "Remove when the MUSA fused kernels cover interleaved and "
            "context-parallel packed sequences, or when upstream Megatron selects the "
            "unfused kernel for those combinations itself."
        ),
    ),
)
