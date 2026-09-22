"""Attention implementation selection: plain functions, fixed candidates.

The framework patch parses the upstream contract
(signature, masks, layouts, return shape) and captures the original handle;
this module only checks input metadata against the *measured* capability
windows and calls the chosen implementation.  No Binding/Provider
registration, no generic selector protocol.

Candidate implementations (operator-local names, not global IDs):

- ``mudnn``       native MUSA flash SDPA path, reached through the captured
                  original TE forward (MT-TE hard-codes use_flash_attention)
- ``mate``        mate's TileLang flash kernels (``flash_attn_varlen_func``)
- ``te_unfused``  TE's own UnfusedDotProductAttention (reference backend)
- ``flash_attn``  declared slot: measured GQA backward instability (closed)
- ``torch_sdpa_math`` declared slot: math-path forcing unverified (closed)

Measured windows (muDNN v3107 / MT-TE 2.0.0 / torch_musa 2.7.1 / mate 0.2.6)
are documented in docs/PATCH_LEDGER.md; they are stack-specific
evidence, not a universal MUSA ranking.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .._config import ATTENTION_IMPLEMENTATIONS

__all__ = [
    "DECLARED_IMPLEMENTATIONS",
    "DEFAULT_CANDIDATE_ORDER",
    "AttentionMeta",
    "Implementation",
    "NoCompatibleImplementation",
    "resolve_candidates",
    "select_and_run",
    "IMPLEMENTATIONS",
]

#: All implementation names this call site knows about (config validation).
DECLARED_IMPLEMENTATIONS = tuple(sorted(ATTENTION_IMPLEMENTATIONS))

#: Fixed candidate order for the verified stack; reference implementations
#: join through fallback=reference, not through the default order.
DEFAULT_CANDIDATE_ORDER = ("mudnn", "mate")

# One bounded snapshot per call site; no tensors or unbounded event history.
_LAST_DISPATCH: dict[str, dict[str, Any]] = {}


def dispatch_report() -> dict[str, dict[str, Any]]:
    return {site: dict(record) for site, record in _LAST_DISPATCH.items()}


def _note_dispatch(call_site: str, implementation: str, entry: Any, reasons) -> None:
    _LAST_DISPATCH[call_site] = {
        "implementation": implementation,
        "entry": f"{getattr(entry, '__module__', '')}:{getattr(entry, '__qualname__', type(entry).__name__)}",
        "rejections": tuple(reasons),
    }


#: MuDNN flash window (torch_musa 2.7.1 / MT-TE 2.0.0, muDNN v3107).
_MIN_FLASH_DIM = 64
_MAX_FLASH_DIM = 192
#: Head dims the MuDNN flash *backward* kernel actually serves.
_FLASH_BWD_SAFE_DIMS = frozenset((64, 80, 96, 112, 128, 160))

#: mate TileLang windows, restricted to the shapes MuDNN's backward rejects.
_MATE_EQUAL_DIMS = frozenset((144, 168, 176, 184, 192))
_MATE_MIXED_DIMS = frozenset(((192, 128), (160, 128)))

_FLOAT_DTYPES = ("torch.float16", "torch.bfloat16", "torch.float32", "torch.float64")


class NoCompatibleImplementation(RuntimeError):
    """No candidate satisfied the policy for this call."""

    def __init__(
        self, call_site: str, meta_summary: str, reasons: Sequence[tuple[str, str]]
    ):
        self.call_site = call_site
        self.reasons = tuple(reasons)
        detail = (
            "; ".join(f"{who}: {reason}" for who, reason in self.reasons)
            or "no candidates"
        )
        super().__init__(
            f"no compatible attention implementation for {call_site} "
            f"({meta_summary}); rejections: {detail}"
        )


@dataclass(frozen=True)
class AttentionMeta:
    """Static call metadata the implementations decide on (no tensor reads).

    The framework patch fills this from the live call.  Fields the patch
    cannot express must make the call ineligible upstream instead of being
    silently defaulted.
    """

    dtype: str  # single dtype across q/k/v; mixed -> dtype_mixed
    dtype_mixed: bool
    device_type: str  # "musa" | "cuda" | "cpu"
    plain_tensors: bool
    heads_q: int
    heads_kv: int
    head_dim_qk: int
    head_dim_v: int
    layout: str  # "sbhd" | "bshd" | "thd"
    batch: int
    seq_q: int
    seq_k: int
    packed: bool
    mask_kind: str  # no_mask|causal|padding|padding_causal|causal_bottom_right|padding_causal_bottom_right|arbitrary
    has_attention_mask: bool
    has_attention_bias: bool
    sliding_window: (
        tuple[int, int] | None
    )  # None = unrestricted (TE's (-1,0)/(-1,-1) normalized by the patch)
    softmax_scale: float | None
    has_alibi: bool
    training: bool
    dropout_p: float
    may_require_backward: bool  # checkpoint recompute counts
    deterministic: bool
    cp_size: int
    fp8: bool
    extra_outputs: bool

    def summary(self) -> str:
        return (
            f"dtype={self.dtype}, device={self.device_type}, layout={self.layout}, "
            f"Hq={self.heads_q}, Hkv={self.heads_kv}, dqk={self.head_dim_qk}, "
            f"dv={self.head_dim_v}, sq={self.seq_q}, sk={self.seq_k}, "
            f"packed={self.packed}, mask={self.mask_kind}, bias={self.has_attention_bias}, "
            f"dropout={self.dropout_p}, may_bwd={self.may_require_backward}"
        )


@dataclass
class Implementation:
    """One candidate: name, role, metadata check, lazy loader, runner.

    ``supports`` reads metadata only: no kernel launches, no collectives,
    no RNG consumption, no tensor sync.
    """

    name: str
    role: str  # "accelerated" | "reference"
    supports_fn: Callable[[AttentionMeta], tuple[bool, str]]
    load_fn: Callable[[], tuple[Any, str | None]]
    run_fn: Callable[[Any, Any], Any]
    _loaded: Any = field(default=None, repr=False)
    _load_error: str | None = field(default=None, repr=False)

    def ensure_loaded(self) -> tuple[Any, str | None]:
        """Safe-init load (once): (payload, rejection_reason)."""
        if self._loaded is not None or self._load_error is not None:
            return self._loaded, self._load_error
        # Only loaders may classify a known absence. Unexpected import/ABI
        # failures propagate; they must never silently select another kernel.
        self._loaded, self._load_error = self.load_fn()
        return self._loaded, self._load_error

    def supports(self, meta: AttentionMeta) -> tuple[bool, str]:
        return self.supports_fn(meta)


# ---------------------------------------------------------------------------
# Capability windows (measured; see module docstring)
# ---------------------------------------------------------------------------


def _mudnn_supports(meta: AttentionMeta) -> tuple[bool, str]:
    if meta.device_type != "musa":
        return False, "device is not musa"
    if not meta.plain_tensors:
        return False, "non-plain tensors"
    if meta.dtype_mixed or meta.dtype not in ("torch.float16", "torch.bfloat16"):
        return False, f"dtype {meta.dtype} (flash asserts fp16/bf16)"
    if meta.packed:
        return False, "packed THD drops cu_seqlens in the flash wrapper"
    if meta.dropout_p != 0.0:
        return False, "flash traps when dropout_p > 0"
    if meta.fp8 or meta.has_alibi or meta.extra_outputs:
        return False, "fp8/alibi/extra outputs stay upstream"
    dims = {meta.head_dim_qk, meta.head_dim_v}
    if any(not _MIN_FLASH_DIM <= dim <= _MAX_FLASH_DIM for dim in dims):
        return False, f"head dims {sorted(dims)} outside forward window"
    if meta.may_require_backward:
        if len(dims) > 1:
            return False, "mixed qk/v head dims fail the MuDNN flash backward"
        if not dims <= _FLASH_BWD_SAFE_DIMS:
            return False, f"head dim {sorted(dims)[0]} fails the MuDNN flash backward"
    return True, ""


def _mate_supports(meta: AttentionMeta) -> tuple[bool, str]:
    if meta.device_type != "musa":
        return False, "device is not musa"
    if not meta.plain_tensors:
        return False, "non-plain tensors"
    if meta.dtype_mixed or meta.dtype not in ("torch.float16", "torch.bfloat16"):
        return False, f"dtype {meta.dtype}"
    if meta.packed:
        return False, "mate varlen exposes no padded-THD physical offsets"
    if meta.dropout_p != 0.0:
        return False, "mate has no dropout"
    if meta.has_attention_bias:
        return False, "no attention bias"
    if meta.mask_kind not in ("no_mask", "causal", "causal_bottom_right"):
        return False, f"mask {meta.mask_kind}"
    if meta.mask_kind == "causal_bottom_right" and meta.seq_q != meta.seq_k:
        return False, "bottom-right causal only when sq == sk"
    if meta.heads_kv and meta.heads_q % meta.heads_kv:
        return False, "q heads not divisible by kv heads"
    if meta.sliding_window not in (None, (-1, -1)):
        # TE's (-1, 0) causal encoding must be normalized by the patch.
        return False, "no sliding-window support (un-normalized window encoding)"
    if meta.fp8 or meta.has_alibi or meta.extra_outputs:
        return False, "fp8/alibi/extra outputs"
    dqk, dv = meta.head_dim_qk, meta.head_dim_v
    if dqk == dv:
        if dqk not in _MATE_EQUAL_DIMS:
            return False, f"equal head dim {dqk} outside mate's verified set"
    elif (dqk, dv) not in _MATE_MIXED_DIMS:
        return False, f"mixed head dims ({dqk}, {dv}) outside mate's verified set"
    return True, ""


def _te_unfused_supports(meta: AttentionMeta) -> tuple[bool, str]:
    if meta.device_type != "musa":
        return False, "device is not musa"
    if not meta.plain_tensors:
        return False, "non-plain tensors"
    if meta.dtype_mixed or meta.dtype not in _FLOAT_DTYPES:
        return False, f"dtype {meta.dtype}"
    if meta.packed:
        return False, "packed THD is sliced by the patch before reaching impls"
    if meta.layout not in ("sbhd", "bshd"):
        return False, f"layout {meta.layout}"
    if meta.mask_kind not in (
        "no_mask",
        "causal",
        "padding",
        "padding_causal",
        "causal_bottom_right",
        "padding_causal_bottom_right",
        "arbitrary",
    ):
        return False, f"mask {meta.mask_kind}"
    if meta.cp_size > 1:
        return False, "context parallel stays upstream"
    if meta.fp8 or meta.has_alibi or meta.extra_outputs:
        return False, "fp8/alibi/extra outputs"
    if (
        meta.mask_kind in ("padding", "padding_causal", "arbitrary")
        and not meta.has_attention_mask
    ):
        return False, f"{meta.mask_kind} mask requires an attention_mask tensor"
    return True, ""


def _closed_slot(reason: str) -> Callable[[AttentionMeta], tuple[bool, str]]:
    def supports(meta: AttentionMeta) -> tuple[bool, str]:
        return False, reason

    return supports


# ---------------------------------------------------------------------------
# Loaders & runners
# ---------------------------------------------------------------------------


def _load_mate() -> tuple[Any, str | None]:
    """Load mate lazily; only an absent top-level package is an optional miss."""
    try:
        import mate
    except ModuleNotFoundError as exc:
        if exc.name != "mate":
            raise
        return None, "mate package not installed"
    fn = getattr(mate, "flash_attn_varlen_func", None)
    if fn is None:
        return None, "mate exposes no flash_attn_varlen_func"
    return fn, None


def _run_mate(fn, call) -> Any:
    """mate consumes/produces [batch, seq, heads, dim]; sbhd is transposed
    in and the output flattened back to TE's dense contract."""
    module = call.module
    layout = call.meta.layout
    if layout == "bshd":
        q, k, v = call.query, call.key, call.value
    else:
        q, k, v = (t.transpose(0, 1) for t in (call.query, call.key, call.value))
    out = fn(
        q,
        k,
        v,
        causal="causal" in call.meta.mask_kind,
        softmax_scale=getattr(module, "softmax_scale", None),
        deterministic=bool(getattr(module, "deterministic", False)),
    )
    if layout == "bshd":
        return out.reshape(out.shape[0], out.shape[1], -1)
    out = out.transpose(0, 1)
    return out.reshape(out.shape[0], out.shape[1], -1)


def _run_mudnn(_payload, call) -> Any:
    """Native flash via the captured original handle (never a re-dispatch)."""
    return call.original(
        call.module,
        call.query,
        call.key,
        call.value,
        call.attention_mask,
        call.attn_mask_type,
        attention_bias=call.attention_bias,
        packed_seq_params=None,
        num_splits=None,
    )


def _te_padding_mask(attention_mask, sq, sk, attention_type="self"):
    """Normalize Megatron padding masks to the shapes TE's get_full_mask
    consumes; None for masks this call site does not claim."""
    import torch

    if attention_mask is None:
        return None

    def shaped(mask):
        if mask.dim() == 2:
            return mask[:, None, None, :]
        if mask.dim() == 3 and mask.shape[1] == 1:
            return mask[:, None, :, :]
        if mask.dim() == 4 and mask.shape[1:3] == (1, 1):
            return mask
        return None

    masks = attention_mask if isinstance(attention_mask, tuple) else (attention_mask,)
    if len(masks) not in (1, 2) or any(
        not isinstance(m, torch.Tensor) or m.dtype != torch.bool for m in masks
    ):
        return None
    shaped_masks = tuple(shaped(m) for m in masks)
    if any(m is None for m in shaped_masks):
        return None
    if attention_type == "cross":
        if len(shaped_masks) != 2:
            return None
        q_mask, k_mask = shaped_masks
        if (
            q_mask.shape[-1] != sq
            or k_mask.shape[-1] != sk
            or q_mask.shape[0] != k_mask.shape[0]
        ):
            return None
        return q_mask, k_mask
    if attention_type != "self" or sq != sk:
        return None
    if any(m.shape[-1] != sk for m in shaped_masks):
        return None
    if len(shaped_masks) == 2 and not torch.equal(*shaped_masks):
        # One self-attention token mask cannot represent distinct Q/K masks;
        # OR-ing would silently mask additional valid queries/keys.
        return None
    return shaped_masks[0]


def _run_te_unfused(_payload, call) -> Any:
    """TE's own UnfusedDotProductAttention backend (the reference path)."""
    backend = getattr(call.module, "unfused_attention", None)
    if backend is None:
        raise RuntimeError(
            "te_unfused selected but the module has no unfused_attention backend"
        )
    layout = call.meta.layout
    if layout == "bshd":
        sq, sk = call.query.shape[1], call.key.shape[1]
    else:
        sq, sk = call.query.shape[0], call.key.shape[0]
    mask = None
    mask_kind = call.meta.mask_kind
    if "padding" in mask_kind:
        mask = _te_padding_mask(
            call.attention_mask, sq, sk, getattr(backend, "attention_type", "self")
        )
        if mask is None:
            # Unclaimable padding mask: surface a clear error instead of
            # guessing a normalization (never silently mask more tokens).
            raise RuntimeError(
                "attention mask shape is not representable for TE get_full_mask"
            )
    elif mask_kind == "arbitrary":
        mask = call.attention_mask
    # Other mask types drop the tensor exactly like the flash path they
    # replace (Megatron sometimes passes a dummy all-ones mask whose TE
    # semantics -- True=masked -- would blank the whole output).
    return backend(
        call.query,
        call.key,
        call.value,
        qkv_layout=f"{layout}_{layout}_{layout}",
        attn_mask_type=mask_kind,
        attention_mask=mask,
        window_size=getattr(call.module, "window_size", None),
        core_attention_bias_type=(
            "post_scale_bias" if call.attention_bias is not None else "no_bias"
        ),
        core_attention_bias=call.attention_bias,
    )


def _closed_slot_run(message: str) -> Callable[[Any, Any], Any]:
    """A named raiser for closed slots; the function name appears in
    tracebacks instead of ``<lambda>``."""

    def run(_payload, _call) -> Any:
        raise RuntimeError(message)

    return run


IMPLEMENTATIONS: dict[str, Implementation] = {
    "mudnn": Implementation(
        name="mudnn",
        role="accelerated",
        supports_fn=_mudnn_supports,
        load_fn=lambda: (True, None),
        run_fn=_run_mudnn,
    ),
    "mate": Implementation(
        name="mate",
        role="accelerated",
        supports_fn=_mate_supports,
        load_fn=_load_mate,
        run_fn=_run_mate,
    ),
    "te_unfused": Implementation(
        name="te_unfused",
        role="reference",
        supports_fn=_te_unfused_supports,
        load_fn=lambda: (True, None),
        run_fn=_run_te_unfused,
    ),
    "flash_attn": Implementation(
        name="flash_attn",
        role="accelerated",
        supports_fn=_closed_slot(
            "measured (flash-attn 2.6.3 MUSA): dense asserts dropout_p == 0 and "
            "the GQA backward fails unreproducibly across identical fresh "
            "processes; slot closed"
        ),
        load_fn=lambda: (None, "slot closed (measured backward instability)"),
        run_fn=_closed_slot_run("flash_attn slot is closed on this stack"),
    ),
    "torch_sdpa_math": Implementation(
        name="torch_sdpa_math",
        role="reference",
        supports_fn=_closed_slot(
            "math-backend forcing is not verified on this MUSA stack; an SDPA "
            "interface that re-dispatches internally is not a reference"
        ),
        load_fn=lambda: (None, "slot closed (unverified math path)"),
        run_fn=_closed_slot_run("torch_sdpa_math slot is not verified on this stack"),
    ),
}


# ---------------------------------------------------------------------------
# Candidate resolution & selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    kind: str  # "implementation" | "original"
    name: str = ""
    role: str = "accelerated"


def resolve_candidates(
    policy: str,
    implementations: Iterable[str],
    fallback: str,
    *,
    reference_impls: Iterable[str] = ("te_unfused",),
) -> tuple[list[_Candidate], list[tuple[str, str]]]:
    """Full candidate order for one call site under the frozen policy.

    Returns (candidates, pre_rejections).  prefer-listed names this call
    site cannot use are recorded and skipped; force/upstream always pair
    with fallback=error (already normalized by the config layer).
    """
    pre_rejections: list[tuple[str, str]] = []
    implementations = tuple(implementations)
    if policy == "upstream":
        return [_Candidate("original")], pre_rejections
    if policy == "force":
        (name,) = implementations
        if name not in IMPLEMENTATIONS:
            raise NoCompatibleImplementation(
                "?", "", [(name, "unknown implementation")]
            )
        return [
            _Candidate("implementation", name, IMPLEMENTATIONS[name].role)
        ], pre_rejections

    candidates: list[_Candidate] = []
    if policy == "prefer":
        for name in implementations:
            if name in IMPLEMENTATIONS:
                candidates.append(
                    _Candidate("implementation", name, IMPLEMENTATIONS[name].role)
                )
            else:
                pre_rejections.append((name, "unknown implementation"))
        for name in DEFAULT_CANDIDATE_ORDER:
            if all(candidate.name != name for candidate in candidates):
                candidates.append(_Candidate("implementation", name))
    else:  # auto
        candidates = [
            _Candidate("implementation", name) for name in DEFAULT_CANDIDATE_ORDER
        ]

    if fallback == "reference":
        for name in reference_impls:
            if all(candidate.name != name for candidate in candidates):
                candidates.append(_Candidate("implementation", name, "reference"))
    elif fallback == "upstream":
        candidates.append(_Candidate("original"))
    # fallback == "error": nothing appended.
    return candidates, pre_rejections


def select_and_run(
    call_site: str,
    candidates: Sequence[_Candidate],
    pre_rejections: Sequence[tuple[str, str]],
    meta: AttentionMeta,
    call: Any,
    original_supports: Callable[[Any], tuple[bool, str]],
    call_original: Callable[[Any], Any],
) -> Any:
    """Run the first candidate whose measured window covers this call.

    Once an implementation starts running, its exceptions propagate -- the
    candidate loop is never re-entered.
    """
    reasons: list[tuple[str, str]] = list(pre_rejections)
    for candidate in candidates:
        if candidate.kind == "original":
            supported, reason = original_supports(call)
            if supported:
                result = call_original(call)
                _note_dispatch(
                    call_site,
                    "upstream",
                    getattr(call, "original", call_original),
                    reasons,
                )
                return result
            reasons.append(("original", reason))
            continue
        implementation = IMPLEMENTATIONS[candidate.name]
        supported, reason = implementation.supports(meta)
        if not supported:
            reasons.append((candidate.name, reason))
            continue
        payload, load_error = implementation.ensure_loaded()
        if payload is None:
            reasons.append((candidate.name, load_error or "not loadable"))
            continue
        result = implementation.run_fn(payload, call)
        entry = payload if callable(payload) else implementation.run_fn
        _note_dispatch(call_site, candidate.name, entry, reasons)
        return result
    raise NoCompatibleImplementation(call_site, meta.summary(), reasons)
