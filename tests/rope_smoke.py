"""Hardware worker for test_rope.py; run on a MUSA stack (single process).

Checks the whole chain against a real Megatron: the fused kernels exist after
the patch set is applied, ``apply_rope_fusion`` (argparse's default) is accepted
by ``TransformerConfig``, and the installed kernels agree with Megatron's own
unfused implementation for both the ``sbhd`` and the packed ``thd`` layouts --
forward and backward.  The interleaved demotion is checked too.
"""

import sys
from types import SimpleNamespace

import torch

import training_musa_adaptor as tma

tma.install()

from megatron.core.models.common.embeddings import rope_utils
from megatron.core.transformer.transformer_config import TransformerConfig

# One CP rank: the shim only has to answer size()/rank(), which is all both the
# fused and the unfused paths ask of a process group.
CP_GROUP = SimpleNamespace(size=lambda: 1, rank=lambda: 0)

statuses = {record["id"]: record["status"] for record in tma.report()["patches"]}
te = sys.modules.get("megatron.core.extensions.transformer_engine")
te_has_kernels = bool(te) and all(
    getattr(te, name, None) is not None
    for name in ("fused_apply_rotary_pos_emb", "fused_apply_rotary_pos_emb_thd")
)
if not te_has_kernels:
    assert statuses["megatron.embeddings.fused-rope.apex"] == "applied", statuses
    assert statuses["megatron.embeddings.fused-rope-thd.apex"] == "applied", statuses
    assert statuses["megatron.embeddings.rope-fusion.unfused-fallback"] == "applied", statuses
assert rope_utils.fused_apply_rotary_pos_emb is not None
assert rope_utils.fused_apply_rotary_pos_emb_thd is not None

config = TransformerConfig(
    num_layers=1,
    hidden_size=64,
    num_attention_heads=4,
    params_dtype=torch.bfloat16,
    bf16=True,
    apply_rope_fusion=True,
)
assert config.apply_rope_fusion is True, "TransformerConfig rejected apply_rope_fusion"
print("ROPE_PASS config")


def through_megatron(t, freqs, cu_seqlens=None, cp_group=CP_GROUP, **config_overrides):
    """Dispatch through Megatron's own (patched) entry point."""
    saved = {key: getattr(config, key) for key in config_overrides}
    for key, value in config_overrides.items():
        setattr(config, key, value)
    try:
        return rope_utils.apply_rotary_pos_emb(
            t, freqs, config=config, cu_seqlens=cu_seqlens, cp_group=cp_group
        )
    finally:
        for key, value in saved.items():
            setattr(config, key, value)


def compare(name, fused, reference, atol, rtol):
    torch.testing.assert_close(fused.float(), reference.float(), atol=atol, rtol=rtol)
    print(f"ROPE_PASS {name}")


def gradients(fn, t):
    out = fn(t)
    out.sum().backward()
    return t.grad


# --- sbhd ------------------------------------------------------------------
torch.manual_seed(0)
seq, batch, heads, dim = 8, 2, 4, 64
x = torch.randn(seq, batch, heads, dim, device="musa", dtype=torch.bfloat16)
freqs = torch.randn(seq, 1, 1, dim, device="musa", dtype=torch.float32)

fused = through_megatron(x, freqs)
reference = through_megatron(x, freqs, apply_rope_fusion=False)
compare("sbhd-forward", fused, reference, atol=0.1, rtol=0.05)

grad_fused = gradients(
    lambda t: through_megatron(t, freqs), x.detach().clone().requires_grad_(True)
)
grad_reference = gradients(
    lambda t: through_megatron(t, freqs, apply_rope_fusion=False),
    x.detach().clone().requires_grad_(True),
)
compare("sbhd-backward", grad_fused, grad_reference, atol=0.1, rtol=0.05)

# --- thd (padded layout, as the fused kernels require) ----------------------
max_seq, lengths = 8, [8, 5, 2]
tokens = max_seq * len(lengths)
packed = torch.randn(tokens, heads, dim, device="musa", dtype=torch.bfloat16)
cu_seqlens = torch.arange(0, tokens + 1, max_seq, dtype=torch.int32, device="musa")
packed_freqs = torch.randn(max_seq, 1, 1, dim, device="musa", dtype=torch.float32)

fused_thd = through_megatron(packed, packed_freqs, cu_seqlens=cu_seqlens)
reference_thd = through_megatron(
    packed, packed_freqs, cu_seqlens=cu_seqlens, apply_rope_fusion=False
)
compare("thd-forward", fused_thd, reference_thd, atol=0.1, rtol=0.05)

grad_fused = gradients(
    lambda t: through_megatron(t, packed_freqs, cu_seqlens=cu_seqlens),
    packed.detach().clone().requires_grad_(True),
)
grad_reference = gradients(
    lambda t: through_megatron(t, packed_freqs, cu_seqlens=cu_seqlens, apply_rope_fusion=False),
    packed.detach().clone().requires_grad_(True),
)
compare("thd-backward", grad_fused, grad_reference, atol=0.1, rtol=0.05)

# --- interleaved is demoted to the unfused kernel, not fused ----------------
interleaved = through_megatron(x, freqs, rotary_interleaved=True)
reference_interleaved = through_megatron(x, freqs, rotary_interleaved=True, apply_rope_fusion=False)
compare("interleaved-demotion", interleaved, reference_interleaved, atol=0.1, rtol=0.05)
assert config.rotary_interleaved is False, "the config flag must be left untouched"
assert config.apply_rope_fusion is True, "the config flag must be left untouched"

# Packed/interleaved needs the same demotion; apex THD has no such variant.
packed_interleaved = through_megatron(
    packed, packed_freqs, cu_seqlens=cu_seqlens, rotary_interleaved=True
)
packed_reference = through_megatron(
    packed,
    packed_freqs,
    cu_seqlens=cu_seqlens,
    rotary_interleaved=True,
    apply_rope_fusion=False,
)
compare("thd-interleaved-forward", packed_interleaved, packed_reference, atol=0.1, rtol=0.05)
actual_grad = gradients(
    lambda t: through_megatron(t, packed_freqs, cu_seqlens=cu_seqlens, rotary_interleaved=True),
    packed.detach().clone().requires_grad_(True),
)
expected_grad = gradients(
    lambda t: through_megatron(
        t, packed_freqs, cu_seqlens=cu_seqlens, rotary_interleaved=True, apply_rope_fusion=False
    ),
    packed.detach().clone().requires_grad_(True),
)
compare("thd-interleaved-backward", actual_grad, expected_grad, atol=0.1, rtol=0.05)
