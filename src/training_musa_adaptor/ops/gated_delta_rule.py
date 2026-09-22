"""Gated delta rule dispatcher: TileLang where supported, FLA everywhere else.

Shared by two framework call sites:

- ``patches/megatron/ssm.py``     -- Megatron Core's FLA binding
- ``patches/mcore_bridge/ssm.py`` -- mcore-bridge's own re-imported binding

The TileLang stack is version-bound (tilelang-musa + torch-kernels must match
the torch_musa/musa_toolkits release) and its kernels JIT-compile on first
use per head count and specialization (minutes, cached in ~/.tilelang;
pre-warm once -- concurrent multi-rank compiles into the shared cache have
crashed runs).
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any

__all__ = ["make_tilelang_dispatcher", "MARKER"]

logger = logging.getLogger("training_musa_adaptor")

#: Marks the wrappers this dispatcher installs (one per binding).
MARKER = "_training_musa_adaptor_tk_gdn"

#: One warning per reason: a silent per-call demotion would hide a dead fast
#: path, but a per-layer warning would flood the training log.
_warned: set[str] = set()

#: Set on the first successful dispatch, so a training log answers "did the
#: TileLang path run?" without profiler work.
_dispatch_noted = False

#: fla's positional parameters (fla 0.5.x has no ``head_first``; the layout is
#: fixed at ``[B, T, H, D]``, also torch-kernels' ``head_first=False``).
_POSITIONAL = ("q", "k", "v", "g", "beta")

#: Keyword arguments the torch-kernels front door understands and this
#: dispatcher is allowed to forward.  Everything else (fla-only activations
#: such as ``use_beta_sigmoid_in_kernel``, CP contexts, ...) stays with FLA.
_KNOWN_KWARGS = frozenset(
    _POSITIONAL
    + (
        "scale",
        "initial_state",
        "output_final_state",
        "use_qk_l2norm_in_kernel",
        "cu_seqlens",
    )
)


def _warn_once(key: str, message: str, *args: Any) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(message, *args)


def _tilelang_stack() -> (
    tuple[Callable[..., Any], Callable[..., bool], Callable[..., bool]] | None
):
    """torch-kernels' GDN front door plus its own shape guards, or ``None``.

    Called from ``replace`` while the target module is being patched, i.e. in
    a process that already runs a real MUSA torch.  Importing torch-kernels
    here (and the TileLang backend it resolves lazily) keeps the patch
    modules importable on machines without the stack. A missing package
    declines; errors inside an installed backend remain visible.
    """
    try:
        from torch_kernels.attention import gated_delta_net as front_door
        from torch_kernels.attention.gated_delta_net import is_backend_available
        from torch_kernels.attention.tilelang.flash_linear_attention.gdn_shapes import (
            gdn_dense_supported,
            gdn_varlen_supported,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "torch_kernels":
            raise
        _warn_once(
            "torch-kernels-missing",
            "training-musa-adaptor: torch-kernels is unavailable (%s: %s); the "
            "gated delta rule keeps running on flash-linear-attention's kernels.",
            type(exc).__name__,
            exc,
        )
        return None
    if not is_backend_available("tilelang"):
        _warn_once(
            "torch-kernels-no-tilelang",
            "training-musa-adaptor: torch-kernels has no tilelang gated-delta-rule "
            "backend registered; keeping flash-linear-attention.",
        )
        return None
    return front_door, gdn_dense_supported, gdn_varlen_supported


def _normalized_call(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any] | None:
    """fla-shaped arguments as a plain dict, or ``None`` when not translatable.

    The wrapper must stay signature-transparent for FLA: unknown keywords,
    duplicate bindings or missing tensors all take the fallback, because the
    front door would either reject them or attach different semantics.
    """
    if len(args) > len(_POSITIONAL):
        return None
    call = dict(zip(_POSITIONAL, args, strict=False))  # fewer positionals is legitimate
    for key, value in kwargs.items():
        if key not in _KNOWN_KWARGS or key in call:
            return None
        call[key] = value
    if any(call.get(name) is None for name in _POSITIONAL):
        return None
    return call


def _supported_by_tilelang(
    call: dict[str, Any],
    dense_supported: Callable[..., bool],
    varlen_supported: Callable[..., bool],
) -> bool:
    """Whether this exact call is inside the TileLang kernels' audited envelope.

    Device/dtype are the kernels' own hard requirements (bf16 q/k/v/beta, fp32
    log-space decay, MUSA tensors); the shape predicates are the operator's
    documented authority.  Reading ``cu_seqlens`` to the host costs one small
    sync per packed call; the per-sequence two-chunk precondition cannot be
    checked from metadata alone.
    """
    import torch

    tensors = [call[name] for name in _POSITIONAL]
    if not all(torch.is_tensor(t) for t in tensors):
        return False
    q, k, v, g, beta = tensors
    device = q.device
    if device.type != "musa" or any(t.device != device for t in (k, v, g, beta)):
        return False
    if any(t.dtype is not torch.bfloat16 for t in (q, k, v, beta)):
        return False
    if g.dtype not in (torch.float32, torch.float64):
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or g.ndim != 3 or beta.ndim != 3:
        return False
    # The kernel shape predicates describe one normalized shape. They do
    # not validate agreement between inputs or the GVA head expansion.
    if q.shape != k.shape or any(t.shape[:2] != q.shape[:2] for t in (v, g, beta)):
        return False
    query_heads, heads = q.shape[2], v.shape[2]
    if query_heads < 1 or heads < 1 or heads % query_heads:
        return False
    if g.shape != beta.shape or g.shape[2] != heads:
        return False

    scale = call.get("scale")
    if scale is not None:
        if (
            isinstance(scale, torch.Tensor)
            or not isinstance(scale, (int, float))
            or not scale > 0
        ):
            return False

    initial_state = call.get("initial_state")
    if initial_state is not None and (
        not torch.is_tensor(initial_state)
        or initial_state.device != device
        or initial_state.shape != (q.shape[0], heads, k.shape[3], v.shape[3])
    ):
        return False

    heads, key_dim, value_dim = v.shape[2], k.shape[3], v.shape[3]
    cu_seqlens = call.get("cu_seqlens")
    if cu_seqlens is None:
        batch, seq = q.shape[0], q.shape[1]
        # Defensive restatement of the relayout constraint the shape guard
        # leaves implicit: odd head counts outside one tile would fail at
        # kernel compile time, and FLA serves them fine.
        if heads > 8 and heads % 8:
            return False
        return dense_supported(batch, seq, heads, key_dim, value_dim)
    if (
        not torch.is_tensor(cu_seqlens)
        or cu_seqlens.ndim != 1
        or cu_seqlens.shape[0] < 2
        or cu_seqlens.dtype not in (torch.int32, torch.int64)
        or q.shape[0] != 1
    ):
        return False
    # The host read is already required by the per-sequence shape guard.
    # Inspect endpoints too, and compute differences here instead of
    # launching a device kernel and silently truncating floating offsets.
    offsets = cu_seqlens.detach().cpu().tolist()
    if offsets[0] != 0 or offsets[-1] != q.shape[1]:
        return False
    lengths = [end - start for start, end in zip(offsets, offsets[1:], strict=False)]
    if any(length < 0 for length in lengths):
        return False
    return varlen_supported(lengths, heads, key_dim, value_dim)


def make_tilelang_dispatcher(original: Any) -> Any:
    """Build the same-signature dispatcher for one FLA binding, or decline.

    Returns ``None`` when FLA is missing (Megatron refuses GDN layers then
    anyway), when the binding is already marked, or when the TileLang stack
    is unavailable -- every declined case keeps the upstream binding.
    """
    if original is None or getattr(original, MARKER, False):
        return None
    if original.__module__ == __name__:
        return None
    stack = _tilelang_stack()
    if stack is None:
        return None
    front_door, dense_supported, varlen_supported = stack

    @functools.wraps(original)
    def chunk_gated_delta_rule(*args: Any, **kwargs: Any):
        global _dispatch_noted
        call = _normalized_call(args, kwargs)
        if call is None or not _supported_by_tilelang(
            call, dense_supported, varlen_supported
        ):
            return original(*args, **kwargs)
        if not _dispatch_noted:
            _dispatch_noted = True
            q = call["q"]
            logger.info(
                "training-musa-adaptor: chunked gated delta rule dispatched to "
                "torch-kernels tilelang (B=%d, S=%d, H=%d, D=%d); "
                "TRAINING_MUSA_ADAPTOR_DISABLE restores flash-linear-attention.",
                q.shape[0],
                q.shape[1],
                q.shape[2],
                q.shape[3],
            )
        return front_door(
            call["q"],
            call["k"],
            call["v"],
            call["g"],
            call["beta"],
            backend="tilelang",
            scale=call.get("scale"),
            initial_state=call.get("initial_state"),
            output_final_state=call.get("output_final_state", False),
            use_qk_l2norm_in_kernel=call.get("use_qk_l2norm_in_kernel", False),
            cu_seqlens=call.get("cu_seqlens"),
        )

    setattr(chunk_gated_delta_rule, MARKER, True)
    return chunk_gated_delta_rule
