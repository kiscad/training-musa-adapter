"""Hardware worker for test_ssm.py; run on a MUSA stack (single process).

Checks the whole chain on real kernels: after the patch set is applied, both
GDN bindings (Megatron core and, when installed, mcore-bridge) carry the
dispatcher; a supported call is answered bit-identically by torch-kernels'
TileLang front door (proving the dispatch actually happened), agrees with FLA
and Megatron's fp32 reference to bf16 tolerance forward and backward, short
sequences and fla-only keywords are demoted to FLA bit-identically, and
``GDN_TILELANG=0`` (fresh process) restores FLA's binding.
"""

from __future__ import annotations

import os
import subprocess
import sys

import torch

import training_musa_adaptor as tma

tma.install()

import fla.modules.l2norm  # noqa: E402  (after the patch activation, like training)
from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_gdn  # noqa: E402
from megatron.core.ssm import gated_delta_net as core_gdn  # noqa: E402
from torch_kernels.attention import gated_delta_net as tk_gdn  # noqa: E402

statuses = {record["id"]: record["status"] for record in tma.report()["patches"]}
assert statuses["megatron.ssm.gated-delta-rule.tilelang"] == "applied", statuses
assert getattr(core_gdn.chunk_gated_delta_rule, "_training_musa_adaptor_tk_gdn", False), statuses
print("SSM_PASS core-binding")

DEV = "musa"


def model_shaped(batch, seq, heads, key_dim=128, value_dim=128, seed=0):
    """Megatron's exact pre-kernel inputs: l2-normalized q/k, fp32 decay."""
    generator = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(batch, seq, heads, key_dim, generator=generator, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(batch, seq, heads, key_dim, generator=generator, device=DEV, dtype=torch.bfloat16)
    q = fla.modules.l2norm.l2norm(q.contiguous()).contiguous()
    k = fla.modules.l2norm.l2norm(k.contiguous()).contiguous()
    v = torch.randn(batch, seq, heads, value_dim, generator=generator, device=DEV, dtype=torch.bfloat16)
    g = -torch.rand(batch, seq, heads, generator=generator, device=DEV, dtype=torch.float32) * 2.0
    beta = torch.rand(batch, seq, heads, generator=generator, device=DEV, dtype=torch.bfloat16)
    return q, k, v, g, beta


def through_patched(t, **kwargs):
    return core_gdn.chunk_gated_delta_rule(*t, **kwargs)


def rel_diff(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def assert_close(name, a, b, atol=5e-2, rtol=5e-2):
    torch.testing.assert_close(a.float(), b.float(), atol=atol, rtol=rtol)
    print(f"SSM_PASS {name} (relL2 {rel_diff(a, b):.2e})")


KW = dict(initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False)

# The mcore-bridge binding (the forward ms-swift's --bridge_backend
# mcore-bridge actually executes) must dispatch too.
bridge_status = statuses.get("mcore_bridge.ssm.gated-delta-rule.tilelang")
if bridge_status == "applied":
    import mcore_bridge.model.modules.gated_delta_net as bridge_gdn  # noqa: E402

    assert getattr(bridge_gdn.chunk_gated_delta_rule, "_training_musa_adaptor_tk_gdn", False)
    q, k, v, g, beta = model_shaped(1, 512, 8, seed=3)
    with torch.no_grad():
        via_bridge = bridge_gdn.chunk_gated_delta_rule(q, k, v, g=g, beta=beta, **KW)[0]
        direct_tk = tk_gdn(q, k, v, g, beta, backend="tilelang", **KW)[0]
    assert torch.equal(via_bridge, direct_tk), "the bridge binding must run torch-kernels too"
    print("SSM_PASS bridge-binding")
else:  # pragma: no cover - wheel-only stacks have no mcore-bridge
    print(f"SSM_SKIP bridge-binding (status {bridge_status!r})")

for heads in (8, 16):
    q, k, v, g, beta = model_shaped(1, 512, heads, seed=heads)
    with torch.no_grad():
        patched = through_patched((q, k, v, g, beta), **KW)[0]
        direct_tk = tk_gdn(q, k, v, g, beta, backend="tilelang", **KW)[0]
        fla_out = fla_gdn(q, k, v, g=g, beta=beta, **KW)[0]
    # Bit-identical to the TileLang front door proves the dispatch ran; FLA
    # would differ at the last-bit level even when numerically equivalent.
    assert torch.equal(patched, direct_tk), "the patched binding must run torch-kernels"
    assert_close(f"forward-vs-fla-h{heads}", patched, fla_out, atol=0.1, rtol=0.05)

# Forward + backward against FLA and Megatron's fp32 reference.
import megatron.core.ssm.gated_delta_net as mgdn  # noqa: E402

q, k, v, g, beta = model_shaped(1, 512, 8, seed=7)
leaf = [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)]
patched_out = through_patched(leaf, **KW)[0]
(patched_out.float().pow(2).mean()).backward()
patched_grads = [x.grad.clone() for x in leaf]

for x in leaf:
    x.grad = None
fla_out = fla_gdn(*leaf, **KW)[0]
(fla_out.float().pow(2).mean()).backward()
fla_grads = [x.grad for x in leaf]

with torch.no_grad():
    ref_out = mgdn.torch_chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=False)[0]

assert_close("forward-vs-reference", patched_out, ref_out, atol=0.1, rtol=0.05)
assert_close("forward-vs-fla", patched_out, fla_out, atol=0.1, rtol=0.05)
for name, got, want in zip(("dq", "dk", "dv", "dg", "dbeta"), patched_grads, fla_grads):
    assert_close(f"dgrad-{name}-vs-fla", got, want, atol=0.1, rtol=0.05)
assert all(torch.isfinite(x).all() for x in patched_grads), "gradients must stay finite"

# Demotions: short sequences and fla-only keywords run FLA bit-identically.
short = model_shaped(1, 64, 8, seed=11)
with torch.no_grad():
    demoted = through_patched(short, **KW)[0]
    fla_short = fla_gdn(*short, **KW)[0]
assert torch.equal(demoted, fla_short), "short sequences must fall back to fla bit-identically"
print("SSM_PASS demotion-short-sequence")

q, k, v, g, beta = model_shaped(1, 512, 8, seed=13)
with torch.no_grad():
    fla_only = through_patched(
        (q, k, v, g, beta), use_beta_sigmoid_in_kernel=False, **KW
    )[0]
    fla_same = fla_gdn(q, k, v, g=g, beta=beta, use_beta_sigmoid_in_kernel=False, **KW)[0]
assert torch.equal(fla_only, fla_same), "fla-only keywords must fall back bit-identically"
print("SSM_PASS demotion-fla-only-keyword")

# DISABLE in a fresh process restores FLA's binding (v2.0 switch semantics).
env = dict(os.environ)
env["TRAINING_MUSA_ADAPTOR_DISABLE"] = "megatron.ssm.gated-delta-rule.tilelang"
probe = subprocess.run(
    [
        sys.executable,
        "-c",
        "import training_musa_adaptor as tma;"
        "tma.install();"
        "from megatron.core.ssm import gated_delta_net as g;"
        "st = {r['id']: r['status'] for r in tma.report()['patches']};"
        "assert st['megatron.ssm.gated-delta-rule.tilelang'] == 'skipped', st;"
        "assert not getattr(g.chunk_gated_delta_rule, '_training_musa_adaptor_tk_gdn', False);"
        "print('SSM_PASS switch-off')",
    ],
    env=env,
    capture_output=True,
    text=True,
    timeout=600,
)
assert probe.returncode == 0, probe.stdout + probe.stderr
print(probe.stdout.strip())
print("SSM_PASS all")
