"""Gated delta rule on MUSA: torch-kernels' TileLang dispatch behind the FLA seam.

The unit rounds are torch-free of MUSA hardware: the dispatcher's decisions are
exercised against stub torch-kernels modules and CPU tensors wearing a fake
``musa`` device, so the whole contract runs on any torch build.  The real
kernels are exercised by ``gdn_smoke.py`` on a MUSA stack.
"""

from __future__ import annotations

import torch

from training_musa_adaptor.ops import gated_delta_rule as _gdn_ops
from training_musa_adaptor.patches import mcore_bridge as _bridge
from training_musa_adaptor.patches.megatron import ssm as _ssm

CORE_GDN = "megatron.core.ssm.gated_delta_net"
BRIDGE_GDN = "mcore_bridge.model.modules.gated_delta_net"

CORE_ID = "megatron.ssm.gated-delta-rule.tilelang"
BRIDGE_ID = "mcore_bridge.ssm.gated-delta-rule.tilelang"

#: The clauses of torch-kernels' audited envelope the stubs mirror (the real
#: guards stay the authority; here they only have to behave like a predicate).
_MIN_SEQ = 128
_HEAD_DIMS = (64, 128)


class Recorder:
    """Callable stand-in for fla's kernel or the torch-kernels front door."""

    def __init__(self, result="fla-result"):
        self.result = result
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class FakeDevice:
    """Duck-typed device: the dispatcher only reads ``.type`` and compares."""

    def __init__(self, type_):
        self.type = type_

    def __eq__(self, other):
        if isinstance(other, torch.device):
            return other.type == self.type
        return isinstance(other, FakeDevice) and other.type == self.type

    def __hash__(self):
        return hash((FakeDevice, self.type))

    def __repr__(self):
        return f"FakeDevice({self.type!r})"


class FakeMusaTensor(torch.Tensor):
    """A CPU tensor that reports the device the dispatcher must select on."""

    @property
    def device(self):  # noqa: D102 - overrides the C-level property
        return FakeDevice("musa")


def fake_musa(t):
    return t.as_subclass(FakeMusaTensor)


def gdn_inputs(batch=1, seq=512, heads=8, dim=128, dtype=torch.bfloat16, g_dtype=torch.float32):
    """fla-shaped tensors on the fake musa device (B, S, H, D / B, S, H)."""
    return (
        fake_musa(torch.randn(batch, seq, heads, dim, dtype=dtype)),
        fake_musa(torch.randn(batch, seq, heads, dim, dtype=dtype)),
        fake_musa(torch.randn(batch, seq, heads, dim, dtype=dtype)),
        fake_musa(torch.rand(batch, seq, heads, dtype=g_dtype)),
        fake_musa(torch.rand(batch, seq, heads, dtype=dtype)),
    )


class StubStack:
    """torch-kernels stand-ins plus a record of every shape-guard consult."""

    def __init__(self, stub_module, *, front_door=None, dense=True, varlen=True):
        self.front_door = front_door or Recorder("tk-result")
        self.guard_calls = {"dense": [], "varlen": []}

        def dense_supported(B, S, H, DK, DV, chunk_size=64):
            self.guard_calls["dense"].append((B, S, H, DK, DV))
            return dense and DK == DV and DK in _HEAD_DIMS and S >= _MIN_SEQ

        def varlen_supported(lengths, H, DK, DV, chunk_size=64):
            lengths = list(lengths)
            self.guard_calls["varlen"].append((lengths, H, DK, DV))
            return varlen and DK == DV and DK in _HEAD_DIMS and all(
                length >= _MIN_SEQ for length in lengths
            )

        stub_module("torch_kernels", __path__=[])
        self.attention = stub_module("torch_kernels.attention", gated_delta_net=self.front_door)
        self.gdn_module = stub_module(
            "torch_kernels.attention.gated_delta_net",
            gated_delta_net=self.front_door,
            is_backend_available=lambda name: name == "tilelang",
        )
        tilelang = stub_module("torch_kernels.attention.tilelang", __path__=[])
        flash_linear_attention = stub_module(
            "torch_kernels.attention.tilelang.flash_linear_attention", __path__=[]
        )
        flash_linear_attention.gdn_shapes = stub_module(
            "torch_kernels.attention.tilelang.flash_linear_attention.gdn_shapes",
            gdn_dense_supported=dense_supported,
            gdn_varlen_supported=varlen_supported,
        )
        tilelang.flash_linear_attention = flash_linear_attention


def install(engine, stub_module, *, core_original=None, bridge_original=None, stack=True, **stack_kwargs):
    """Stub the two GDN bindings (and optionally torch-kernels) for the engine."""
    core_original = core_original or Recorder("fla-result")
    bridge_original = bridge_original or Recorder("bridge-fla-result")
    core = stub_module(CORE_GDN, chunk_gated_delta_rule=core_original)
    bridge = stub_module(BRIDGE_GDN, chunk_gated_delta_rule=bridge_original)
    stub_stack = StubStack(stub_module, **stack_kwargs) if stack else None
    return core, bridge, core_original, bridge_original, stub_stack


def call_kwargs(g, beta, **overrides):
    kwargs = {"g": g, "beta": beta, "initial_state": None, "output_final_state": False}
    kwargs.update(overrides)
    return kwargs


# --- argument normalization (pure logic, no torch) --------------------------


def test_unknown_keyword_stays_with_fla():
    assert _gdn_ops._normalized_call(("q", "k", "v", "g", "beta"), {"use_beta_sigmoid_in_kernel": True}) is None


def test_positional_and_keyword_forms_normalize():
    assert _gdn_ops._normalized_call(("q", "k", "v"), {"g": "g", "beta": "b", "scale": 0.5}) == {
        "q": "q",
        "k": "k",
        "v": "v",
        "g": "g",
        "beta": "b",
        "scale": 0.5,
    }
    assert _gdn_ops._normalized_call(("q", "k", "v", "g", "beta", "extra"), {}) is None
    assert _gdn_ops._normalized_call(("q", "k", "v"), {"g": "g"}) is None  # beta missing
    assert _gdn_ops._normalized_call((), {"q": "q"}) is None  # fla requires all five


def test_duplicate_binding_stays_with_fla():
    assert _gdn_ops._normalized_call(("q", "k"), {"k": "k2", "v": "v", "g": "g", "beta": "b"}) is None


# --- selection --------------------------------------------------------------


def test_supported_call_dispatches_to_tilelang(engine, stub_module):
    core, _, core_original, _, stack = install(engine, stub_module)
    engine.register(_ssm.PATCHES + _bridge.PATCHES)
    engine.install()

    assert core.chunk_gated_delta_rule is not core_original
    assert getattr(core.chunk_gated_delta_rule, _gdn_ops.MARKER, False)

    q, k, v, g, beta = gdn_inputs()
    result = core.chunk_gated_delta_rule(q, k, v, **call_kwargs(g, beta))
    assert result == "tk-result"
    assert stack.front_door.calls, "the supported call must reach the front door"
    args, kwargs = stack.front_door.calls[-1]
    assert args == (q, k, v, g, beta)
    assert kwargs["backend"] == "tilelang"
    assert kwargs["scale"] is None
    assert kwargs["initial_state"] is None
    assert kwargs["output_final_state"] is False
    assert kwargs["use_qk_l2norm_in_kernel"] is False
    assert kwargs["cu_seqlens"] is None
    assert stack.guard_calls["dense"] == [(1, 512, 8, 128, 128)]
    assert core_original.calls == [], "the fallback must not run for a dispatched call"


def test_scale_and_state_are_forwarded(engine, stub_module):
    core, _, _, _, stack = install(engine, stub_module)
    engine.register(_ssm.PATCHES + _bridge.PATCHES)
    engine.install()

    q, k, v, g, beta = gdn_inputs()
    state = fake_musa(torch.zeros(1, 8, 128, 128, dtype=torch.bfloat16))
    core.chunk_gated_delta_rule(
        q, k, v, g, beta, scale=0.25, initial_state=state, output_final_state=True
    )
    _, kwargs = stack.front_door.calls[-1]
    assert kwargs["scale"] == 0.25
    assert kwargs["initial_state"] is state
    assert kwargs["output_final_state"] is True


def test_cpu_and_cuda_tensors_stay_with_fla(engine, stub_module):
    core, _, core_original, _, stack = install(engine, stub_module)
    engine.register(_ssm.PATCHES + _bridge.PATCHES)
    engine.install()

    q = torch.randn(1, 512, 8, 128, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.randn(1, 512, 8, dtype=torch.float32)
    beta = torch.rand(1, 512, 8, dtype=torch.bfloat16)
    core.chunk_gated_delta_rule(q, k, v, g=g, beta=beta)
    assert core_original.calls == [((q, k, v), {"g": g, "beta": beta})]
    assert stack.front_door.calls == []


def test_fla_only_keywords_stay_with_fla(engine, stub_module):
    core, _, core_original, _, stack = install(engine, stub_module)
    engine.register(_ssm.PATCHES + _bridge.PATCHES)
    engine.install()

    q, k, v, g, beta = gdn_inputs()
    core.chunk_gated_delta_rule(q, k, v, g=g, beta=beta, use_beta_sigmoid_in_kernel=True)
    assert core_original.calls == [((q, k, v), {"g": g, "beta": beta, "use_beta_sigmoid_in_kernel": True})]
    assert stack.front_door.calls == []


def test_unsupported_dtypes_stay_with_fla(engine, stub_module):
    core, _, core_original, _, stack = install(engine, stub_module)
    engine.register(_ssm.PATCHES + _bridge.PATCHES)
    engine.install()

    q, k, v, g, beta = gdn_inputs(dtype=torch.float16)
    core.chunk_gated_delta_rule(q, k, v, g=g, beta=beta)
    assert len(core_original.calls) == 1, "fp16 activations are outside the TileLang envelope"

    q, k, v, g, beta = gdn_inputs(g_dtype=torch.bfloat16)
    core.chunk_gated_delta_rule(q, k, v, g=g, beta=beta)
    assert len(core_original.calls) == 2, "g must be fp32 log-space decay"
    assert stack.front_door.calls == []


def test_shape_guard_is_consulted_per_call(engine, stub_module):
    core, _, core_original, _, stack = install(engine, stub_module)
    engine.register(_ssm.PATCHES + _bridge.PATCHES)
    engine.install()

    # Short sequences and head dims outside the audited tiers are FLA's calls.
    for batch, seq, heads, dim in ((1, 64, 8, 128), (1, 512, 8, 96), (1, 512, 12, 128)):
        q, k, v, g, beta = gdn_inputs(batch, seq, heads, dim)
        core.chunk_gated_delta_rule(q, k, v, g=g, beta=beta)
    assert len(core_original.calls) == 3
    assert stack.front_door.calls == []
    assert (1, 512, 12, 128) not in [
        shape for shape in stack.guard_calls["dense"]
    ], "the relayout head-divisibility guard must demote H=12 before asking the operator"


def test_divisible_head_counts_are_not_demoted(engine, stub_module):
    core, _, core_original, _, stack = install(engine, stub_module)
    engine.register(_ssm.PATCHES + _bridge.PATCHES)
    engine.install()

    q, k, v, g, beta = gdn_inputs(1, 512, 16, 128)
    core.chunk_gated_delta_rule(q, k, v, g=g, beta=beta)
    assert stack.front_door.calls and not core_original.calls


def test_packed_calls_check_per_sequence_lengths(engine, stub_module):
    core, _, core_original, _, stack = install(engine, stub_module)
    engine.register(_ssm.PATCHES + _bridge.PATCHES)
    engine.install()

    q, k, v, g, beta = gdn_inputs()
    cu_seqlens = fake_musa(torch.tensor([0, 256, 512], dtype=torch.int32))
    core.chunk_gated_delta_rule(q, k, v, g=g, beta=beta, cu_seqlens=cu_seqlens)
    assert stack.front_door.calls, "a packed batch every sequence can serve must dispatch"
    assert stack.front_door.calls[-1][1]["cu_seqlens"] is cu_seqlens
    assert stack.guard_calls["varlen"] == [([256, 256], 8, 128, 128)]

    short = fake_musa(torch.tensor([0, 64, 512], dtype=torch.int32))
    core.chunk_gated_delta_rule(q, k, v, g=g, beta=beta, cu_seqlens=short)
    assert core_original.calls, "a one-chunk sequence must fall back to fla"


# --- patch lifecycle --------------------------------------------------------



def test_broken_torch_kernels_stack_declines(engine, stub_module):
    # A ``torch_kernels.attention`` without the front door (editable installs
    # bypass the parent's __path__, so break the attribute, not the package):
    # the dispatch probe must decline the patch instead of breaking the first
    # training step.
    core, bridge, core_original, bridge_original, _ = install(engine, stub_module, stack=False)
    stub_module("torch_kernels", __path__=[])
    stub_module("torch_kernels.attention", flash_attention=object())
    engine.register(_ssm.PATCHES + _bridge.PATCHES)
    engine.install()

    assert core.chunk_gated_delta_rule is core_original
    assert bridge.chunk_gated_delta_rule is bridge_original
    assert all(record["status"] == "skipped" for record in engine.report()["patches"])


def test_missing_flas_binding_declines(engine, stub_module):
    core = stub_module(CORE_GDN, chunk_gated_delta_rule=None)
    bridge = stub_module(BRIDGE_GDN, chunk_gated_delta_rule=None)
    StubStack(stub_module)
    engine.register(_ssm.PATCHES + _bridge.PATCHES)
    engine.install()
    assert core.chunk_gated_delta_rule is None
    assert bridge.chunk_gated_delta_rule is None


def test_patches_are_independently_selectable(engine, stub_module, monkeypatch):
    core, bridge, core_original, bridge_original, _ = install(engine, stub_module)
    monkeypatch.setenv("TRAINING_MUSA_ADAPTOR_ONLY", CORE_ID)
    engine.register(_ssm.PATCHES + _bridge.PATCHES)
    engine.install()

    assert core.chunk_gated_delta_rule is not core_original
    assert bridge.chunk_gated_delta_rule is bridge_original
    statuses = {record["id"]: record["status"] for record in engine.report()["patches"]}
    assert statuses[CORE_ID] == "applied"
    assert statuses[BRIDGE_ID] == "skipped"


def test_reapply_does_not_stack_wrappers(engine, stub_module):
    core, _, core_original, _, _ = install(engine, stub_module)
    engine.register(_ssm.PATCHES + _bridge.PATCHES)
    engine.install()
    first = core.chunk_gated_delta_rule
    engine.install()
    assert core.chunk_gated_delta_rule is first

    engine.uninstall()
    assert core.chunk_gated_delta_rule is core_original
    assert getattr(core_original, _gdn_ops.MARKER, False) is False


def test_fla_fallback_receives_the_original_arguments(engine, stub_module):
    core, _, core_original, _, _ = install(engine, stub_module)
    engine.register(_ssm.PATCHES + _bridge.PATCHES)
    engine.install()

    q, k, v, g, beta = gdn_inputs(1, 32, 8, 128)  # below the two-chunk floor
    kwargs = call_kwargs(g, beta, use_qk_l2norm_in_kernel=False)
    core.chunk_gated_delta_rule(q, k, v, **kwargs)
    assert core_original.calls == [((q, k, v), kwargs)]


def test_gdn_hardware_smoke():
    """Real-kernel GDN validation on MUSA (TileLang dispatch, numerics,
    demotion and switch-off), ported worker from megatron-musa-patch."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    import pytest

    if os.environ.get("TMA_RUN_INTEGRATION") != "1":
        pytest.skip("set TMA_RUN_INTEGRATION=1 with a working Megatron/MUSA stack")
    from tests.conftest import integration_env

    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("gdn_smoke.py"))],
        env=integration_env(),
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SSM_PASS all" in result.stdout
