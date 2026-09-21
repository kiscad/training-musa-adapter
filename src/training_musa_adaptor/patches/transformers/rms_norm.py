"""Fused RMSNorm for transformers' Qwen3-VL text stack on MUSA.

ms-swift trains and evaluates Hugging Face ``transformers`` models directly on
MUSA, and the Qwen3-VL text tower normalizes through
``Qwen3VLTextRMSNorm.forward`` three to four times per decoder layer (input,
post-attention, final norm, plus the per-head q/k norms). Upstream implements
the op as a chain of small kernels -- fp32 cast, ``pow(2)``, ``mean``, eps
add, ``rsqrt``, multiply, cast back, weight multiply -- so every call pays
several kernel launches where one fused kernel would do. On torch_musa the
fused ATen op ``torch.rms_norm`` measured 8.1x faster forward and 5.4x faster
forward+backward than the upstream chain (bf16, 8192x4096, MTT S5000; measured
by megatron-musa-patch, see docs/PATCHES.md there).

This is a kernel-selection patch, not a crash repair: the upstream chain is
numerically correct, and the fused kernel differs from it only in rounding
(the fused kernel keeps fp32 through the weight multiply; the upstream chain
rounds to the input dtype first). ``@use_kernel_forward_from_hub("RMSNorm")``
is inert unless the caller opts into the ``kernels`` library, in which case
upstream's own swap wins at model init.

Migrated from megatron-musa-patch ``patches/_transformers.py`` (rev a1090de)
to the design-doc §4.1 target writing: compared with the old wrapper this
adds the plain-Tensor guard (tensor subclasses delegate), the MUSA-device
guard (non-MUSA calls delegate instead of relying on torch.rms_norm to
error), the weight/activation same-device guard, and the fp16/bf16 dtype
restriction (fp32 calls stay on the upstream chain -- the fused win is a
reduced-precision win and fp32 delegation is bit-identical to upstream).
The old per-patch env switch (MEGATRON_MUSA_PATCH_QWEN3VL_RMS_NORM) is
retired: v2.0 configuration disables patches by ID
(TRAINING_MUSA_ADAPTOR_DISABLE=transformers.qwen3-vl.text-rms-norm.fused-torch).
"""

from __future__ import annotations

import functools
from typing import Any

from ..._engine import AttrPatch

__all__ = ["PATCHES"]


def replace_rms_norm(original: Any) -> Any:
    """Run the norm through one ``torch.rms_norm`` call instead of small ops.

    Only the fully matched reduced-precision MUSA path is claimed; everything
    the fused op does not cover delegates to the original forward so
    upstream's semantics stay exact:

    - extra positional/keyword arguments (an extended upstream signature is
      upstream's business, not this patch's re-derivation of it);
    - tensor subclasses (their dispatch belongs to the subclass);
    - non-MUSA activations (CPU/CUDA tensors keep the upstream chain);
    - weight on a different device than the activation;
    - weight/activation dtype mismatch -- upstream's final multiply
      (``weight * hidden_states.to(input_dtype)``) promotes when the
      parameter dtype differs (e.g. fp32 weights under autocast), while
      ``torch.rms_norm`` keeps the input dtype;
    - fp32 activations -- the fused kernel's win is a reduced-precision win;
      delegating keeps fp32 outputs bit-identical to upstream.
    """
    import torch  # the factory runs when the target module is ready

    @functools.wraps(original)
    def forward(self, hidden_states, *args, **kwargs):
        if (
            args
            or kwargs
            or type(hidden_states) is not torch.Tensor
            or hidden_states.device.type != "musa"
            or self.weight.device != hidden_states.device
            or hidden_states.dtype != self.weight.dtype
            or hidden_states.dtype not in (torch.float16, torch.bfloat16)
        ):
            return original(self, hidden_states, *args, **kwargs)
        return torch.rms_norm(
            hidden_states,
            (hidden_states.shape[-1],),
            self.weight,
            self.variance_epsilon,
        )

    return forward


PATCHES = (
    AttrPatch(
        id="transformers.qwen3-vl.text-rms-norm.fused-torch",
        target="transformers.models.qwen3_vl.modeling_qwen3_vl:Qwen3VLTextRMSNorm.forward",
        replace=replace_rms_norm,
        version_gates=("transformers >=4.57",),
        rationale=(
            "ms-swift's transformers-native runs execute the Qwen3-VL text "
            "tower's Qwen3VLTextRMSNorm.forward as a chain of six-plus small "
            "kernels (fp32 cast, pow, mean, eps add, rsqrt, multiply, cast "
            "back, weight multiply) three to four times per decoder layer; on "
            "MUSA every kernel in the chain is a separate launch, which makes "
            "the norm a measurable share of non-GEMM step time. This is a "
            "kernel-selection patch, not a crash repair -- the upstream chain "
            "is correct."
        ),
        strategy=(
            "Replace only the class's forward: the fully matched path (plain "
            "torch.Tensor on a MUSA device, weight on the same device, "
            "matching fp16/bf16 dtype, no extra arguments) issues a single "
            "torch.rms_norm over the last dimension (fused forward kernel, "
            "ATen autograd-decomposed backward); every other call -- extra "
            "arguments, tensor subclasses, non-MUSA activations, cross-device "
            "weight, dtype mismatch (e.g. fp32 weights under autocast, where "
            "upstream's final multiply promotes the output) and fp32 "
            "activations -- delegates to the original forward so its "
            "semantics stay exact. Parameters, state dict, the "
            "variance_epsilon attribute and upstream's kernels-library "
            "opt-in are untouched; torch is imported inside the factory. "
            "Measured on MTT S5000 (bf16, 8192x4096): forward 8.1x, "
            "forward+backward 5.4x versus the upstream chain; bf16/fp16 "
            "outputs agree with the upstream chain to within ~1-2 ulp of "
            "the input dtype (the fused kernel rounds once, after the "
            "weight multiply). "
            "TRAINING_MUSA_ADAPTOR_DISABLE=transformers.qwen3-vl.text-rms-norm.fused-torch "
            "keeps the upstream chain."
        ),
        upstream=(
            "huggingface/transformers "
            "transformers/models/qwen3_vl/modeling_qwen3_vl.py:"
            "Qwen3VLTextRMSNorm.forward"
        ),
        remove_when=(
            "Review on transformers or torch_musa upgrades; remove when "
            "upstream fuses the op itself or torch_musa's eager small-op "
            "kernels make the chain competitive, verified by re-running "
            "tests/test_transformers_rms_norm.py and "
            "tests/integration/test_rms_norm_musa.py with this patch "
            "disabled."
        ),
    ),
)
