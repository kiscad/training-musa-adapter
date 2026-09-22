"""Plain TP=1 delayed weight gradients for the MT-TE 2.0 API gap.

Keep Megatron/TE construction and state dicts. Only the unavailable delayed
forward/backward contract is implemented; no version predicate is changed.
"""

from __future__ import annotations

import copy
import functools
from collections import deque

from ..._engine import AttrPatch
from ...backends import musa_available

__all__ = ["PATCHES"]


def _delayed_init(original):
    if not musa_available():
        return None

    @functools.wraps(original)
    def init(self, *args, config, **kwargs):
        if not config.delay_wgrad_compute:
            return original(self, *args, config=config, **kwargs)
        if getattr(config, "fp8", None) or getattr(config, "fp4", None):
            raise NotImplementedError(
                "MUSA delayed wgrad fallback requires non-quantized parameters"
            )
        # A private construction view bypasses only the unavailable constructor
        # argument. The real config and delayed schedule remain enabled.
        construction = copy.copy(config)
        construction.delay_wgrad_compute = False
        original(self, *args, config=construction, **kwargs)
        self.config = config
        if self.tp_size != 1:
            raise NotImplementedError(
                "MUSA delayed wgrad fallback currently requires TP=1"
            )
        self._musa_pending_wgrads = deque()

    return init


def _linear(x, weight, bias, owner):
    import torch

    class DelayedLinear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp, w, b):
            ctx.save_for_backward(inp, w)
            ctx.has_bias = b is not None
            ctx.weight_parameter = w
            return torch.nn.functional.linear(
                inp, w.to(inp.dtype), b.to(inp.dtype) if b is not None else None
            )

        @staticmethod
        def backward(ctx, grad):
            inp, w = ctx.saved_tensors
            dx = grad @ w.to(grad.dtype) if ctx.needs_input_grad[0] else None
            dy = grad.reshape(-1, grad.shape[-1])
            if ctx.needs_input_grad[1]:
                owner._musa_pending_wgrads.append(
                    (ctx.weight_parameter, inp.reshape(-1, inp.shape[-1]), dy)
                )
            db = dy.sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None
            return dx, None, db

    return DelayedLinear.apply(x, weight, bias)


def _delayed_forward(original):
    if not musa_available():
        return None

    @functools.wraps(original)
    def forward(self, x, *args, **kwargs):
        if not hasattr(self, "_musa_pending_wgrads"):
            return original(self, x, *args, **kwargs)
        import torch
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

        if (
            FP8GlobalStateManager.is_fp8_enabled()
            or FP8GlobalStateManager.is_fp8_calibration()
        ):
            raise NotImplementedError(
                "MUSA delayed wgrad fallback does not implement FP8"
            )
        grouped = hasattr(self, "num_gemms")
        with self.prepare_forward(x, num_gemms=self.num_gemms if grouped else 1) as inp:
            inp = inp.to(self.activation_dtype)
            if grouped:
                splits = args[0] if args else kwargs["m_splits"]
                chunks = inp.reshape(-1, inp.shape[-1]).split(splits)
                weights = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
                biases = [getattr(self, f"bias{i}") for i in range(self.num_gemms)]
            else:
                chunks, weights, biases = [inp], [self.weight], [self.bias]
            outputs = [
                _linear(
                    chunk,
                    w,
                    b if self.use_bias and not self.te_return_bias else None,
                    self,
                )
                for chunk, w, b in zip(chunks, weights, biases, strict=True)
            ]
            out = (
                torch.cat(outputs, 0).view(*inp.shape[:-1], -1)
                if grouped
                else outputs[0]
            )
        self.is_first_microbatch = False
        bias = (
            (
                [b.to(self.activation_dtype) for b in biases]
                if grouped
                else biases[0].to(self.activation_dtype)
            )
            if self.te_return_bias
            else None
        )
        return out, bias

    return forward


def _delayed_backward(original):
    @functools.wraps(original)
    def backward_dw(self):
        if not hasattr(self, "_musa_pending_wgrads"):
            return original(self)
        import torch

        with torch.no_grad():
            while self._musa_pending_wgrads:
                w, inp, dy = self._musa_pending_wgrads.popleft()
                grad = dy.float().T @ inp.float()
                if self.fuse_wgrad_accumulation:
                    w.main_grad.add_(grad.to(w.main_grad.dtype))
                    if hasattr(w, "grad_added_to_main_grad"):
                        w.grad_added_to_main_grad = True
                elif w.grad is None:
                    w.grad = grad.to(w.dtype)
                else:
                    w.grad.add_(grad.to(w.dtype))

    return backward_dw


PATCHES = tuple(
    AttrPatch(
        id=f"megatron.te.{name.lower()}.delayed-{operation}",
        rebind_prefixes=("megatron",),
        target=f"megatron.core.extensions.transformer_engine:{name}.{method}",
        replace=factory,
        requires=(
            (
                f"megatron.te.{name.lower()}.delayed-forward",
                f"megatron.te.{name.lower()}.delayed-backward",
            )
            if operation == "init"
            else ()
        ),
        # delay_wgrad_compute and TELinear/TEGroupedLinear.backward_dw exist
        # from core_v0.13.0 and are unchanged through core_v0.19.0, the
        # newest release line in the checkout; MT-TE 2.0 is the fork whose
        # missing delayed contract this fallback supplies.
        version_gates=(
            "megatron-core >=0.13,<0.20",
            "transformer_engine >=2.0,<2.1",
        ),
        rationale="MT-TE 2.0 lacks delay_wgrad_compute and backward_dw; offload overlap construction fails.",
        strategy=(
            "Keep original construction with a private config view only when both execution "
            "wrappers are installed. Implement ordinary TP=1 linear autograd with deferred "
            "weight GEMMs drained by backward_dw. Saved tensors use existing offload hooks. "
            "Parameters, bias gradients and state dicts stay intact. FP8 and TP>1 are explicitly "
            "unsupported; per-expert GEMMs cost throughput and FP32 scratch."
        ),
        upstream="Megatron-LM 55ac7082 megatron/core/extensions/transformer_engine.py:"
        + name,
        remove_when="Remove after native MT-TE delayed wgrad passes offload overlap and gradient timing tests.",
    )
    for name in ("TELinear", "TEGroupedLinear")
    for operation, method, factory in (
        ("init", "__init__", _delayed_init),
        ("forward", "forward", _delayed_forward),
        ("backward", "backward_dw", _delayed_backward),
    )
)
