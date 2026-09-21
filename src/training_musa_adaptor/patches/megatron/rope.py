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

Moore Threads' apex fork ships equivalent fused kernels, which is also what the
pre-0.16 MUSA patch set used.  This module binds them where upstream left
``None``; it never overrides a kernel Transformer Engine actually provided.
"""

from __future__ import annotations

import functools
import logging
from copy import copy
from typing import Any, Optional

from ..._engine import AttrPatch

__all__ = ["PATCHES"]

logger = logging.getLogger("megatron_musa_patch")

#: The one module all three patches share: Megatron's rope dispatch and its two
#: optional fused kernels.
_ROPE_UTILS = "megatron.core.models.common.embeddings.rope_utils"

#: One warning per reason: the fused path runs on every attention layer.
_warned: set[str] = set()

#: Marks the wrappers this module installs, so the dispatcher can tell an
#: apex kernel from one Transformer Engine provided (same convention as the
#: LayerNorm fallback's ``_megatron_musa_patch_fallback``).
_MARKER = "_megatron_musa_patch_apex_rope"


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


def _apex_kernels() -> Optional[tuple[Any, Any]]:
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
    except Exception as exc:  # ImportError, or OSError from a broken extension
        _warn_once(
            "apex-missing",
            "megatron-musa-patch: apex's fused RoPE is unavailable (%s: %s); "
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


def _apex_fused_thd(original: Any) -> Any:
    """Upstream-shaped ``thd`` fused RoPE backed by apex's kernel."""
    if original is not None:
        return None
    kernels = _apex_kernels()
    if kernels is None:
        return None
    _, apex_fused_thd = kernels

    @functools.wraps(apex_fused_thd)
    def fused_apply_rotary_pos_emb_thd(t, cu_seqlens, freqs, cp_size: int = 1, cp_rank: int = 0):
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


def _context_parallel_group() -> Any:
    """Upstream's own fallback for a caller that passes no ``cp_group``."""
    from megatron.core import parallel_state

    return parallel_state.get_context_parallel_group()


def _unfusable_reason(
    config: Any, cu_seqlens: Any, cp_group: Any, *, apex_bshd: bool, apex_thd: bool
) -> str:
    """Why the installed kernels cannot serve this call, or ``""`` when they can.

    Only apex's limits are reported: a kernel Transformer Engine provided is not
    demoted here, whatever it supports.
    """
    if cu_seqlens is None:
        if apex_bshd and getattr(config, "rotary_interleaved", False):
            return "interleaved"
        return ""
    if not apex_thd:
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
    def apply_rotary_pos_emb(t, freqs, config, cu_seqlens=None, mscale: float = 1.0, cp_group=None):
        reason = ""
        if config.apply_rope_fusion:
            reason = _unfusable_reason(
                config,
                cu_seqlens,
                cp_group,
                apex_bshd=_is_apex_kernel(_bound_kernel("fused_apply_rotary_pos_emb")),
                apex_thd=_is_apex_kernel(_bound_kernel("fused_apply_rotary_pos_emb_thd")),
            )
        if not reason and config.apply_rope_fusion and getattr(config, "rotary_interleaved", False):
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
            if extension is not None and kernel is not None and kernel is vars(extension).get(name):
                version_check = vars(extension).get("is_te_min_version")
                if callable(version_check) and not version_check("2.3.0"):
                    reason = "interleaved"
        if not reason:
            return original(
                t, freqs, config=config, cu_seqlens=cu_seqlens, mscale=mscale, cp_group=cp_group
            )
        _warn_once(
            reason,
            "megatron-musa-patch: the installed fused RoPE cannot serve %s on MUSA; "
            "using Megatron's unfused rotary embedding for those layers.",
            (
                "rotary_interleaved models"
                if reason == "interleaved"
                else "packed sequences with context parallel > 1"
            ),
        )
        local_config = copy(config)
        local_config.apply_rope_fusion = False
        return original(
            t, freqs, config=local_config, cu_seqlens=cu_seqlens, mscale=mscale, cp_group=cp_group
        )

    return apply_rotary_pos_emb


PATCHES = (
    AttrPatch(
        id="megatron.embeddings.fused-rope.apex",
        rebind_prefixes=("megatron",),
        target=f"{_ROPE_UTILS}:fused_apply_rotary_pos_emb",
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
            "ROPE_FUSION=0 leaves upstream's verdict untouched."
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
            "to the unfused CP-aware kernel. ROPE_FUSION=0 leaves upstream untouched."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/models/common/embeddings/rope_utils.py",
        remove_when=(
            "Remove together with the sbhd fallback, after MUSA TE supplies the fused "
            "thd kernel and padded-layout packed-sequence parity has been re-validated "
            "on MUSA."
        ),
    ),
    AttrPatch(
        id="megatron.embeddings.rope-fusion.unfused-fallback",
        rebind_prefixes=("megatron",),
        target=f"{_ROPE_UTILS}:apply_rotary_pos_emb",
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
