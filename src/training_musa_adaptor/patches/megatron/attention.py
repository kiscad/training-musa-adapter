"""Megatron attention patch: TEDotProductAttention capability dispatch.

Migrated from megatron-musa-patch ``patches/_attention.py`` (rev a1090de)
with the first-round fixes kept: TE causal-window normalization, mate
mixed head-dim support and the packed-THD/empty-sequence handling.

The patch owns the *framework contract* (signature, masks, layouts, packed
THD slicing, return shape); implementation selection lives in
``ops/attention.py``.  The dispatch exists because MT-TE hard-codes
``use_flash_attention = True`` in
``transformer_engine/musa/pytorch/attention.py``: the flash kernel asserts
fp16/bf16, serves head dims 64..192 in the forward, traps on dropout,
drops cu_seqlens (padded THD -> NaN), and its backward is narrower than
its forward.  Measured on muDNN v3107 / MT-TE 2.0.0 / torch_musa 2.7.1
with Megatron core_v0.16.1.
"""

from __future__ import annotations

import functools
from typing import Any

from ..._compat import module_source_contains
from ..._engine import AttrPatch
from ...backends import musa_available as _musa_live
from ...ops import attention as _ops

__all__ = ["PATCHES"]

_TARGET = "megatron.core.extensions.transformer_engine:TEDotProductAttention.forward"
_PATCH_ID = "megatron.te.attention.capability-dispatch"

_ALLOWED_MASKS = {
    "no_mask",
    "causal",
    "padding",
    "padding_causal",
    "causal_bottom_right",
    "padding_causal_bottom_right",
    "arbitrary",
}

#: Candidates resolved once per process at first use (config is frozen by
#: then); hot path never re-reads config or environment.
_RESOLVED: dict[str, Any] = {}


def _mask_type_name(attn_mask_type) -> str:
    return (getattr(attn_mask_type, "name", None) or str(attn_mask_type)).replace(",", "_")


def _attn_mask_type_value(name):
    """Rebuild the caller's enum-like ``attn_mask_type`` for upstream calls."""
    try:
        from megatron.core.transformer.enums import AttnMaskType

        return getattr(AttnMaskType, name)
    except Exception:
        from types import SimpleNamespace

        return SimpleNamespace(name=name)


def _frozen_selection():
    """Frozen policy for this call site (patch_options override the
    operator-generic attention section)."""
    from ...activation import ENGINE

    config = ENGINE.config.config
    selection = config.options_for(_PATCH_ID) or config.attention()
    if "candidates" not in _RESOLVED:
        candidates, pre_rejections = _ops.resolve_candidates(
            selection.policy, selection.implementations, selection.fallback
        )
        _RESOLVED["candidates"] = candidates
        _RESOLVED["pre_rejections"] = pre_rejections
    return selection, _RESOLVED["candidates"], _RESOLVED["pre_rejections"]


# ---------------------------------------------------------------------------
# Call context (passed to ops implementations)
# ---------------------------------------------------------------------------


class _AttentionCall:
    __slots__ = (
        "module",
        "query",
        "key",
        "value",
        "attention_mask",
        "attn_mask_type",
        "attention_bias",
        "original",
        "meta",
    )

    def __init__(self, module, query, key, value, attention_mask, attn_mask_type,
                 attention_bias, original, meta):
        self.module = module
        self.query = query
        self.key = key
        self.value = value
        self.attention_mask = attention_mask
        self.attn_mask_type = attn_mask_type
        self.attention_bias = attention_bias
        self.original = original
        self.meta = meta


def _effective_dropout(self) -> float:
    """Dropout the flash kernel would have to apply on this call."""
    training = bool(getattr(self, "training", False))
    return float(getattr(self, "attention_dropout", 0.0) or 0.0) if training else 0.0


def _may_require_backward(self, tensors) -> bool:
    """A training-mode module may be recomputed in backward (mcore
    checkpoint re-runs the forward under no_grad); grad-enabled calls with
    grad-requiring inputs count regardless of the flag."""
    import torch

    if bool(getattr(self, "training", False)):
        return True
    return torch.is_grad_enabled() and any(getattr(t, "requires_grad", False) for t in tensors)


def _eligible(self, query, key, value, packed, num_splits, mask_name, attention_mask) -> bool:
    """Do not bypass upstream validation/parallel or quantization protocols."""
    import torch

    if any(type(t) is not torch.Tensor for t in (query, key, value)):
        return False
    if any(
        t.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        for t in (query, key, value)
    ):
        return False
    config = getattr(self, "config", None)
    if any(
        getattr(config, name, False)
        for name in (
            "fp8_dot_product_attention",
            "fp8_multi_head_attention",
            "qk_clip",
            "log_max_attention_logit",
            "apply_query_key_layer_scaling",
        )
    ):
        return False
    if getattr(config, "softmax_type", "vanilla") != "vanilla":
        return False
    if num_splits is not None or getattr(self, "num_splits", None) is not None:
        return False
    window = getattr(self, "window_size", None)
    if window is not None and tuple(window) not in ((-1, -1), (-1, 0)):
        return False
    group = getattr(self, "cp_group", None)
    if packed is not None:
        dynamic_group = getattr(packed, "cp_group", None)
        local_size = getattr(packed, "local_cp_size", None)
        if dynamic_group is not None:
            group = dynamic_group
        elif local_size == 1:
            group = None
        elif local_size is not None:
            return False
    if group is not None and (isinstance(group, (list, tuple)) or group.size() > 1):
        return False
    if group is None and getattr(config, "context_parallel_size", 1) > 1:
        if packed is None or getattr(packed, "local_cp_size", None) != 1:
            return False
    if mask_name not in _ALLOWED_MASKS:
        return False
    if packed is None and attention_mask is None and (
        "padding" in mask_name or mask_name == "arbitrary"
    ):
        # Dense calls cannot express a padding/arbitrary mask without the
        # tensor; packed THD spans carry only valid tokens, so the padding
        # qualifier is dropped per span instead (old-code guard restored).
        return False
    if getattr(self, "window_size", None) == (-1, 0) and "causal" not in mask_name:
        return False
    return True


def _build_meta(self, query, key, value, attention_mask, attention_bias, packed, mask_name, layout) -> _ops.AttentionMeta:
    """Static metadata for the implementations (TE window encoding
    normalized here -- the first-round fix)."""
    import torch

    if layout == "bshd":
        batch, seq_q, seq_k = query.shape[0], query.shape[1], key.shape[1]
    elif layout == "thd":
        batch, seq_q, seq_k = 0, query.shape[0], key.shape[0]
    else:  # sbhd
        batch, seq_q, seq_k = query.shape[1], query.shape[0], key.shape[0]
    dtypes = {str(t.dtype) for t in (query, key, value)}
    window = getattr(self, "window_size", None)
    # TE's check_set_window_size normalizes causal-initialized modules to
    # (-1, 0) and plain modules to (-1, -1) -- even when (-1, -1) was given
    # explicitly.  Inside this patch's domain both encodings mean "no
    # windowing beyond the mask kind" (contradictory combos were already
    # rejected by _eligible), so they are normalized away here.
    if window is not None and tuple(window) in ((-1, -1), (-1, 0)):
        window = None
    group = getattr(self, "cp_group", None)
    if packed is not None and getattr(packed, "cp_group", None) is not None:
        group = getattr(packed, "cp_group")
    cp_size = 1
    if group is not None and not isinstance(group, (list, tuple)):
        cp_size = group.size()
    return _ops.AttentionMeta(
        dtype=str(query.dtype),
        dtype_mixed=len(dtypes) > 1,
        device_type=query.device.type,
        plain_tensors=all(type(t) is torch.Tensor for t in (query, key, value)),
        heads_q=query.shape[-2],
        heads_kv=key.shape[-2],
        head_dim_qk=query.shape[-1],
        head_dim_v=value.shape[-1],
        layout=layout,
        batch=batch,
        seq_q=seq_q,
        seq_k=seq_k,
        packed=packed is not None,
        mask_kind=mask_name,
        has_attention_mask=attention_mask is not None,
        has_attention_bias=attention_bias is not None,
        sliding_window=None if window is None else tuple(window),
        softmax_scale=getattr(self, "softmax_scale", None),
        has_alibi=False,
        training=bool(getattr(self, "training", False)),
        dropout_p=_effective_dropout(self),
        may_require_backward=_may_require_backward(self, (query, key, value)),
        deterministic=bool(getattr(self, "deterministic", False)),
        cp_size=cp_size,
        fp8=bool(getattr(getattr(self, "config", None), "fp8_dot_product_attention", False)),
        extra_outputs=False,
    )


# ---------------------------------------------------------------------------
# Packed THD (per-sequence slicing, empty-sequence gradient edges)
# ---------------------------------------------------------------------------


def _packed_spans(cumulative, padded, total):
    """Read only O(batch) metadata; keep activations and gradients on device."""
    lengths = cumulative.detach().cpu().tolist()
    offsets = lengths if padded is None else padded.detach().cpu().tolist()
    if (
        len(lengths) < 2
        or len(offsets) != len(lengths)
        or lengths[0] != 0
        or offsets[0] != 0
        or offsets[-1] != total
    ):
        raise ValueError("Invalid packed cumulative lengths/physical offsets")
    spans = []
    for i in range(len(lengths) - 1):
        count, capacity = lengths[i + 1] - lengths[i], offsets[i + 1] - offsets[i]
        if count < 0 or capacity < count:
            raise ValueError("Packed lengths must be monotonic and fit padded storage")
        spans.append((offsets[i], count, capacity))
    return spans


def _original_supports(self, call) -> tuple[bool, str]:
    """fallback=upstream: only when the original (hard-coded flash) path is
    confirmed applicable for THIS input."""
    meta = call.meta
    if meta is None:
        return False, "call outside the patch domain"
    return _ops.IMPLEMENTATIONS["mudnn"].supports(meta)


def _dense_dispatch(self, query, key, value, attention_mask, attn_mask_type, attention_bias, layout, original):
    """Run the resolved candidate order for one dense sbhd/bshd call."""
    mask_name = _mask_type_name(attn_mask_type)
    meta = _build_meta(self, query, key, value, attention_mask, attention_bias, None, mask_name, layout)
    selection, candidates, pre_rejections = _frozen_selection()
    if selection.policy == "upstream":
        return original(
            self, query, key, value, attention_mask, attn_mask_type,
            attention_bias=attention_bias, packed_seq_params=None, num_splits=None,
        )
    call = _AttentionCall(self, query, key, value, attention_mask, attn_mask_type, attention_bias, original, meta)
    return _ops.select_and_run(
        _PATCH_ID,
        candidates,
        pre_rejections,
        meta,
        call,
        original_supports=lambda c: _original_supports(self, c),
        call_original=lambda c: original(
            c.module, c.query, c.key, c.value, c.attention_mask, c.attn_mask_type,
            attention_bias=c.attention_bias, packed_seq_params=None, num_splits=None,
        ),
    )


def _packed_forward(self, query, key, value, mask_name, packed, original):
    """Segmented THD via per-sequence vendor calls (ported fix).

    The MUSA flash wrapper drops cu_seqlens and yields NaN once physical
    offsets are padded, so each sequence is dispatched separately in sbhd
    layout through the same dense dispatch.  Empty sequences keep
    zero-valued outputs with gradient edges.
    """
    import torch

    q_spans = _packed_spans(
        packed.cu_seqlens_q, getattr(packed, "cu_seqlens_q_padded", None), query.shape[0]
    )
    k_spans = _packed_spans(
        packed.cu_seqlens_kv, getattr(packed, "cu_seqlens_kv_padded", None), key.shape[0]
    )
    if len(q_spans) != len(k_spans) or key.shape[0] != value.shape[0]:
        raise ValueError("Packed query/key/value sequences must match")
    # Spans hold only valid tokens; the padding qualifier is meaningless per
    # span, causal alignment (incl. rectangular bottom-right) is kept.
    span_mask = mask_name.removeprefix("padding_")
    if span_mask == "padding":
        span_mask = "no_mask"
    outputs = []
    for (qs, nq, capacity), (ks, nk, _) in zip(q_spans, k_spans):
        # clone() detaches the span from packed storage: TE's layout probe
        # rejects non-zero storage offsets, and .contiguous() would be a no-op.
        q, k, v = (
            query[qs : qs + nq].clone(),
            key[ks : ks + nk].clone(),
            value[ks : ks + nk].clone(),
        )
        if nq and nk:
            out = _dense_dispatch(
                self,
                q[:, None],
                k[:, None],
                v[:, None],
                None,
                _attn_mask_type_value(span_mask),
                None,
                "sbhd",
                original,
            )
            out = out.reshape(nq, q.shape[-2], v.shape[-1])
        else:
            # Retain zero gradient edges for empty sequences too.
            out = q.sum(-1, keepdim=True).expand(nq, q.shape[-2], v.shape[-1]) * 0
            out = out + (k.sum() + v.sum()) * 0
        if capacity > nq:
            out = torch.cat((out, out.new_zeros(capacity - nq, out.shape[1], out.shape[2])))
        outputs.append(out)
    # TE's THD contract flattens heads: [total, h * d].
    return torch.cat(outputs, dim=0).reshape(query.shape[0], -1)


# ---------------------------------------------------------------------------
# The patch factory
# ---------------------------------------------------------------------------


def _tedpa_forward(original: Any) -> Any:
    """Wrap TEDotProductAttention.forward, or decline when not applicable.

    The dispatch exists because MT-TE hard-codes use_flash_attention; when
    that source fingerprint is gone the patch declines (a missing marker
    alone does not prove native THD/dropout/backward correctness).
    """
    hardcoded_flash = module_source_contains(
        "transformer_engine.musa.pytorch.attention", "use_flash_attention = True"
    )
    if hardcoded_flash is False:
        return None

    @functools.wraps(original)
    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type,
        attention_bias=None,
        packed_seq_params=None,
        num_splits=None,
    ):
        packed = packed_seq_params
        layout = getattr(packed, "qkv_format", None) or getattr(self, "qkv_format", "sbhd")
        mask_name = _mask_type_name(attn_mask_type)
        eligible = _eligible(self, query, key, value, packed, num_splits, mask_name, attention_mask)
        if eligible and _musa_live() and query.device.type == "musa":
            if (
                layout == "thd"
                and packed is not None
                and mask_name != "arbitrary"
                and attention_mask is None
                and attention_bias is None
            ):
                return _packed_forward(self, query, key, value, mask_name, packed, original)
            if packed is None and layout in ("sbhd", "bshd"):
                return _dense_dispatch(
                    self, query, key, value, attention_mask, attn_mask_type,
                    attention_bias, layout, original,
                )
        return original(
            self,
            query,
            key,
            value,
            attention_mask,
            attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed,
            num_splits=num_splits,
        )

    return forward


PATCHES = (
    AttrPatch(
        id=_PATCH_ID,
        target=_TARGET,
        replace=_tedpa_forward,
        rationale=(
            "MT-TE's DotProductAttention hard-codes use_flash_attention, so "
            "every call reaches the MUSA flash SDPA kernel regardless of "
            "capability: fp16/bf16 assert, head dims 64..192 forward window, "
            "dropout trap, cu_seqlens dropped (padded THD -> NaN), backward "
            "narrower than forward (equal dims {64,80,96,112,128,160} only; "
            "144 and 168..192 and every mixed qk/v pair fail, surfacing as "
            "'MuDNNFlashSDPABwd MUDNN failed'). Measured on muDNN v3107 / "
            "MT-TE 2.0.0 / torch_musa 2.7.1 / flash-attn 2.6.3 with Megatron "
            "core_v0.16.1."
        ),
        strategy=(
            "The patch owns the framework contract (eligibility, masks, "
            "layouts, packed THD slicing, return shape); ops/attention.py "
            "selects the implementation by measured capability windows: "
            "native MuDNN flash inside its forward+backward window, mate's "
            "TileLang flash for the MuDNN-backward-broken shapes, TE's own "
            "unfused backend as the reference.  TE's (-1,0)/(-1,-1) window "
            "encoding is normalized before the implementations see it.  "
            "Packed THD is sliced into per-sequence sbhd calls of the same "
            "dispatch, keeping padded offsets and zero-gradient edges for "
            "empty sequences.  CP, FP8 DPA, special softmax, windowed "
            "attention and max-logit stay upstream.  Policy comes from "
            "[attention] / TRAINING_MUSA_ADAPTOR_ATTN_* / "
            "patch_options.\"megatron.te.attention.capability-dispatch\"."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/core/extensions/transformer_engine.py:"
            "TEDotProductAttention.forward; TransformerEngine pytorch/attention.py"
        ),
        remove_when=(
            "Remove when the MUSA TransformerEngine selects backends by input "
            "capability, its flash kernels accept the full head-dim range and "
            "dropout, and THD honors cu_seqlens: disable this patch id and "
            "re-run transformer/test_attention.py, "
            "test_multi_latent_attention.py and resharding/test_model_swap.py; "
            "delete only if the native path passes."
        ),
    ),
)
