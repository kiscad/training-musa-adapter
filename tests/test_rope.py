"""Fused RoPE on MUSA: apex kernels where Transformer Engine provides none.

The unit rounds are torch-free: ``_rope`` imports torch/apex only inside the
``replace`` callables, and everything below runs against synthetic modules.
The real kernels are exercised by ``rope_smoke.py`` on a MUSA stack.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from training_musa_adaptor.patches.megatron import rope as _rope

ROPE_UTILS = "megatron.core.models.common.embeddings.rope_utils"


class Recorder:
    """Callable stand-in for a kernel or for upstream's dispatcher."""

    def __init__(self, result="result"):
        self.result = result
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class _FakeFreqs:
    """Real-shaped freqs stand-in: reshape is the only contract used."""

    shape = (4, 1, 1, 8)

    def reshape(self, *shape):
        return ("reshaped", shape)


class _FakeMusaTensor:
    """4D musa tensor stand-in: device + slicing is all the wrapper uses."""

    device = SimpleNamespace(type="musa")

    def __getitem__(self, key):
        return _FakeView(f"sliced@{key}")


class _FakeView:
    def __init__(self, tag):
        self.tag = tag
        self.numel = 0  # empty passthrough tail: no concat in the wrapper


def _stub_aten(monkeypatch, result="aten-result"):
    """Make ``_rope._aten_rope_op()`` resolve to a recorded fake op."""
    op = Recorder(result)
    monkeypatch.setattr(_rope, "_aten_rope_op", lambda: op)
    return op


def _no_aten(monkeypatch):
    """Make the torch fused-RoPE probe unavailable (apex fallback path)."""
    monkeypatch.setattr(_rope, "_aten_rope_op", lambda: None)


def _stub_apex(stub_module):
    """Make ``_rope._apex_kernels()`` resolve without a real apex install."""
    stub_module("fused_rotary_positional_embedding")
    bshd, thd = Recorder("bshd-result"), Recorder("thd-result")
    stub_module(
        "apex.transformer.functional",
        fused_apply_rotary_pos_emb=bshd,
        fused_apply_rotary_pos_emb_thd=thd,
    )
    return bshd, thd


def _rope_utils(stub_module, *, bshd=None, thd=None, dispatch=None):
    return stub_module(
        ROPE_UTILS,
        fused_apply_rotary_pos_emb=bshd,
        fused_apply_rotary_pos_emb_thd=thd,
        apply_rotary_pos_emb=dispatch or Recorder("dispatch-result"),
    )


@pytest.fixture(autouse=True)
def _no_stale_warnings():
    """Warnings are once-per-process; keep tests independent of each other."""
    _rope._warned.clear()
    yield
    _rope._warned.clear()


# --- kernel selection -----------------------------------------------------


def test_apex_pair_is_installed_when_the_torch_op_is_unavailable(
    engine, stub_module, monkeypatch
):
    _no_aten(monkeypatch)
    apex_bshd, apex_thd = _stub_apex(stub_module)
    module = _rope_utils(stub_module)

    engine.register(_rope.PATCHES)
    engine.install()

    # The aten binding skips (op unavailable) and the core17 thd variant
    # declines here: within one chain the 0.13-0.16 binding already replaced
    # the symbol, and a kernel-provided value (the variant's own
    # "original is not None" rule) always wins.
    assert [record["status"] for record in engine.report()["patches"]] == [
        "skipped",
        "applied",
        "applied",
        "skipped",
        "applied",
    ]
    assert _rope._is_apex_kernel(module.fused_apply_rotary_pos_emb)
    assert _rope._is_apex_kernel(module.fused_apply_rotary_pos_emb_thd)

    sentinel_t, sentinel_freqs = object(), object()
    assert (
        module.fused_apply_rotary_pos_emb(sentinel_t, sentinel_freqs) == "bshd-result"
    )
    assert apex_bshd.calls == [((sentinel_t, sentinel_freqs, False), {})]
    assert (
        module.fused_apply_rotary_pos_emb(
            sentinel_t, sentinel_freqs, transpose_output_memory=True
        )
        == "bshd-result"
    )
    assert apex_bshd.calls[-1] == ((sentinel_t, sentinel_freqs, True), {})

    cu_seqlens = object()
    assert (
        module.fused_apply_rotary_pos_emb_thd(sentinel_t, cu_seqlens, sentinel_freqs)
        == "thd-result"
    )
    assert apex_thd.calls == [((sentinel_t, cu_seqlens, sentinel_freqs), {})]


def test_aten_binding_is_preferred_over_apex(engine, stub_module, monkeypatch):
    """The torch op wins the shared seam; apex takes over only when absent."""
    _stub_aten(monkeypatch)
    apex_bshd, _ = _stub_apex(stub_module)
    module = _rope_utils(stub_module)

    engine.register(_rope.PATCHES)
    engine.install()

    assert [record["status"] for record in engine.report()["patches"]] == [
        "applied",
        "skipped",
        "applied",
        "skipped",
        "applied",
    ]
    bound = module.fused_apply_rotary_pos_emb
    assert getattr(bound, "_tma_aten_rope", False)
    assert not _rope._is_apex_kernel(bound)
    assert apex_bshd.calls == []  # the torch op won the shared seam


def test_aten_binding_policy_overrides_megatron_wrapper_only(stub_module, monkeypatch):
    """Megatron's own TE-version wrapper (the seam this vendor stack binds)
    is overridden by the torch op; a genuine TE kernel always wins."""
    _stub_aten(monkeypatch)

    megatron_wrapper = Recorder("megatron-te-wrapper")
    megatron_wrapper.__module__ = "megatron.core.extensions.transformer_engine"
    overridden = _rope._aten_fused_bshd(megatron_wrapper)
    assert overridden is not None and overridden is not megatron_wrapper

    te_kernel = Recorder("te-kernel")
    te_kernel.__module__ = "transformer_engine.pytorch.attention.rope"
    assert _rope._aten_fused_bshd(te_kernel) is None  # genuine TE kernel wins


def test_aten_wrapper_refuses_non_musa_inputs(stub_module, monkeypatch):
    """The op has only a torch_musa kernel; the dispatcher demotes non-MUSA
    inputs to the unfused kernel before this wrapper is reached."""
    _stub_aten(monkeypatch)
    wrapper = _rope._aten_fused_bshd(None)
    t_cpu = SimpleNamespace(device=SimpleNamespace(type="cpu"))

    with pytest.raises(NotImplementedError, match="torch_musa kernel"):
        wrapper(t_cpu, _FakeFreqs())


@pytest.mark.musa
def test_aten_rope_matches_megatron_reference_on_musa():
    """MUSA 实测（子进程）：aten::rope vs megatron unfused 参考。

    覆盖 fp32/bf16 × 两种 interleaved 模式、直通拆分（D_in>D_freq）与梯度。
    判据：fp32 max|diff|≤1e-4；bf16 ≤ 输入量级处 1 个 bf16 ULP（内核 fp32
    内部计算、单次舍入，抵消点上比逐步 bf16 参考更精确）。
    在子进程运行：补丁激活与设备内存不属于单测进程。
    """
    import os
    import subprocess

    if not os.environ.get("TMA_RUN_INTEGRATION"):
        pytest.skip("set TMA_RUN_INTEGRATION=1 with a working Megatron/MUSA stack")
    worker = """
import torch, torch_musa
import megatron.core.models.common.embeddings.rope_utils as ru

def meg_freqs(seq, d_rot, interleaved):
    inv = 1.0 / (10000 ** (torch.arange(0, d_rot, 2, dtype=torch.float32,
                                           device="musa") / d_rot))
    ang = torch.arange(seq, dtype=torch.float32, device="musa")[:, None] * inv[None, :]
    emb = (torch.stack((ang.view(-1, 1), ang.view(-1, 1)), dim=-1).view(seq, -1)
           if interleaved else torch.cat((ang, ang), dim=-1))
    return emb[:, None, None, :]

S, B, H, DROT = 512, 2, 8, 128
fails = []
for dtype in (torch.float32, torch.bfloat16):
    atol = 1e-4 if dtype == torch.float32 else 3.2e-2  # 1 bf16 ULP @ |t|≈4
    for interleaved in (False, True):
        freqs4 = meg_freqs(S, DROT, interleaved)
        t = torch.randn(S, B, H, DROT, device="musa", dtype=dtype)
        ref = ru._apply_rotary_pos_emb_bshd(t, freqs4, rotary_interleaved=interleaved)
        act = torch.rope(t, freqs4.squeeze(1).squeeze(1),
                         rotary_interleaved=interleaved, batch_first=False)
        ok = torch.allclose(act.float(), ref.float(), atol=atol, rtol=1e-2)
        print(f"case dtype={dtype} interleaved={interleaved}: "
              f"{'PASS' if ok else 'FAIL'} max|diff|={(act.float()-ref.float()).abs().max().item():.3e}")
        fails.append((dtype, interleaved, ok))
    # 直通拆分（D_in=192 > D_freq=128）
    freqs4 = meg_freqs(S, DROT, False)
    t_wide = torch.randn(S, B, H, 192, device="musa", dtype=dtype)
    ref = ru._apply_rotary_pos_emb_bshd(t_wide, freqs4, rotary_interleaved=False)
    act = torch.cat((torch.rope(t_wide[..., :DROT], freqs4.squeeze(1).squeeze(1),
                                rotary_interleaved=False, batch_first=False),
                     t_wide[..., DROT:]), dim=-1)
    ok = torch.allclose(act.float(), ref.float(), atol=atol, rtol=1e-2)
    print(f"case dtype={dtype} passthrough: {'PASS' if ok else 'FAIL'}")
    fails.append((dtype, "passthrough", ok))

# 梯度（fp32 逐位一致）
freqs4 = meg_freqs(S, DROT, False)
t1 = torch.randn(S, B, H, DROT, device="musa", dtype=torch.float32, requires_grad=True)
t2 = t1.detach().clone().requires_grad_(True)
ru._apply_rotary_pos_emb_bshd(t2, freqs4).sum().backward()
torch.rope(t1, freqs4.squeeze(1).squeeze(1), rotary_interleaved=False,
           batch_first=False).sum().backward()
print("case autograd fp32:", "PASS" if torch.equal(t1.grad, t2.grad) else "FAIL")

if any(not ok for _, _, ok in fails):
    raise SystemExit("ROPE-ATEN-VERIFY FAILED")
print("ROPE-ATEN-VERIFY PASS")
"""
    env = {**os.environ, "CUDA_DEVICE_MAX_CONNECTIONS": "1"}
    result = subprocess.run(
        [sys.executable, "-c", worker],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert "ROPE-ATEN-VERIFY PASS" in result.stdout, result.stdout + result.stderr


def test_dispatcher_keeps_interleaved_on_the_aten_fused_path(
    engine, stub_module, monkeypatch
):
    """interleaved 已在 MUSA 上与参考对数一致：dispatcher 不再降级。"""
    _stub_aten(monkeypatch, result="aten-result")
    _stub_apex(stub_module)
    calls = []

    def dispatch(t, freqs, config, cu_seqlens=None, mscale=1.0, cp_group=None):
        calls.append(config.apply_rope_fusion)
        return "unfused-result"

    module = _rope_utils(stub_module, dispatch=dispatch)
    engine.register(_rope.PATCHES)
    engine.install()
    config = SimpleNamespace(apply_rope_fusion=True, rotary_interleaved=True)
    t_musa = _FakeMusaTensor()

    module.apply_rotary_pos_emb(t_musa, _FakeFreqs(), config=config)
    # no demotion: fusion stays enabled and the call reaches upstream's
    # fused branch (the stub dispatch records the untouched flag)
    assert calls == [True]


def test_dispatcher_demotes_non_musa_inputs_when_aten_bound(
    engine, stub_module, monkeypatch
):
    aten_op = _stub_aten(monkeypatch, result="aten-result")
    _stub_apex(stub_module)
    calls = []

    def dispatch(t, freqs, config, cu_seqlens=None, mscale=1.0, cp_group=None):
        calls.append(config.apply_rope_fusion)
        return "unfused-result"

    module = _rope_utils(stub_module, dispatch=dispatch)
    engine.register(_rope.PATCHES)
    engine.install()
    config = SimpleNamespace(apply_rope_fusion=True, rotary_interleaved=False)

    module.apply_rotary_pos_emb(object(), object(), config=config)
    assert calls == [False]  # the op has only a torch_musa kernel
    assert aten_op.calls == []


def test_bshd_kernel_refuses_interleaved(engine, stub_module, monkeypatch):
    _no_aten(monkeypatch)
    _stub_apex(stub_module)
    module = _rope_utils(stub_module)
    engine.register(_rope.PATCHES)
    engine.install()

    with pytest.raises(NotImplementedError, match="interleaved"):
        module.fused_apply_rotary_pos_emb(object(), object(), interleaved=True)


def test_thd_kernel_refuses_context_parallel(engine, stub_module):
    _stub_apex(stub_module)
    module = _rope_utils(stub_module)
    engine.register(_rope.PATCHES)
    engine.install()

    with pytest.raises(NotImplementedError, match="context-parallel"):
        module.fused_apply_rotary_pos_emb_thd(
            object(), object(), object(), cp_size=2, cp_rank=0
        )


def test_thd_core17_variant_keeps_the_0_17_call_contract(stub_module):
    """core>=0.17 thd variant: accepts the dispatcher's interleaved keyword,
    refuses True exactly like the sbhd kernel, keeps the cp_size=1 refusal.
    A separate patch rather than a wider signature on the 0.13-0.16 one."""
    _stub_apex(stub_module)
    patch = next(
        p
        for p in _rope.PATCHES
        if p.id == "megatron.embeddings.fused-rope-thd.apex.core17"
    )
    assert patch.version_gates == ("megatron-core >=0.17,<0.20",)
    assert patch.replace(Recorder("te-thd")) is None  # a TE-provided kernel always wins

    wrapper = patch.replace(None)
    sentinel_t, cu, freqs = object(), object(), object()
    assert wrapper(sentinel_t, cu, freqs) == "thd-result"
    assert wrapper(sentinel_t, cu, freqs, interleaved=False) == "thd-result"
    with pytest.raises(NotImplementedError, match="interleaved"):
        wrapper(sentinel_t, cu, freqs, interleaved=True)
    with pytest.raises(NotImplementedError, match="context-parallel"):
        wrapper(sentinel_t, cu, freqs, cp_size=2)


def test_thd_gates_tile_core_0_13_through_0_19_without_gaps_or_overlap():
    """The 0.13-0.16 binding and the core17 variant must tile the release
    lines: exactly one of them applies for any core in 0.13..0.19."""
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    gates = {
        p.id: p.version_gates
        for p in _rope.PATCHES
        if p.id.startswith("megatron.embeddings.fused-rope-thd")
    }
    assert set(gates) == {
        "megatron.embeddings.fused-rope-thd.apex",
        "megatron.embeddings.fused-rope-thd.apex.core17",
    }
    for version in (
        "0.13.0",
        "0.14.0",
        "0.15.0",
        "0.16.1",
        "0.17.0",
        "0.18.0",
        "0.19.0",
    ):
        hits = [
            pid
            for pid, specs in gates.items()
            if any(
                SpecifierSet(spec.split(" ", 1)[1], prereleases=True).contains(
                    Version(version)
                )
                for spec in specs
            )
        ]
        assert len(hits) == 1, (version, hits)


def test_transformer_engine_kernels_are_never_replaced(engine, stub_module):
    _stub_apex(stub_module)
    te_bshd, te_thd = Recorder("te-bshd"), Recorder("te-thd")
    te_bshd.__module__ = "transformer_engine.pytorch.attention.rope"
    te_thd.__module__ = "transformer_engine.pytorch.attention.rope"
    module = _rope_utils(stub_module, bshd=te_bshd, thd=te_thd)

    engine.register(_rope.PATCHES)
    engine.install()

    assert [record["status"] for record in engine.report()["patches"]] == [
        "skipped",
        "skipped",
        "skipped",
        "skipped",
        "applied",
    ]
    assert module.fused_apply_rotary_pos_emb is te_bshd
    assert module.fused_apply_rotary_pos_emb_thd is te_thd


def test_kernel_patches_decline_when_no_apex_kernels_are_available(
    engine, stub_module, monkeypatch
):
    monkeypatch.setattr(_rope, "_apex_kernels", lambda: None)
    _no_aten(monkeypatch)
    module = _rope_utils(stub_module)

    engine.register(_rope.PATCHES)
    engine.install()

    assert [record["status"] for record in engine.report()["patches"]] == [
        "skipped",
        "skipped",
        "skipped",
        "skipped",
        "applied",
    ]
    assert module.fused_apply_rotary_pos_emb is None
    assert module.fused_apply_rotary_pos_emb_thd is None


def test_apex_lookup_declines_without_the_compiled_extension(stub_module, monkeypatch):
    bshd, thd = _stub_apex(stub_module)
    assert _rope._apex_kernels() == (bshd, thd)

    monkeypatch.setitem(sys.modules, "fused_rotary_positional_embedding", None)
    assert _rope._apex_kernels() is None


def _dispatcher(engine, stub_module, monkeypatch, *, calls, bshd=None, thd=None):
    """Install the fallback trio and return (module, config).

    The torch fused-RoPE probe is made unavailable so these tests pin the
    apex/dispatcher semantics; the aten binding has dedicated tests.
    """
    _no_aten(monkeypatch)
    _stub_apex(stub_module)

    def dispatch(t, freqs, config, cu_seqlens=None, mscale=1.0, cp_group=None):
        calls.append(config.apply_rope_fusion)
        return "unfused-result"

    module = _rope_utils(stub_module, bshd=bshd, thd=thd, dispatch=dispatch)
    engine.register(_rope.PATCHES)
    engine.install()
    return module


def test_interleaved_configs_are_demoted_and_the_flag_is_restored(
    engine, stub_module, monkeypatch
):
    calls = []
    module = _dispatcher(engine, stub_module, monkeypatch, calls=calls)
    config = SimpleNamespace(apply_rope_fusion=True, rotary_interleaved=True)

    assert (
        module.apply_rotary_pos_emb(object(), object(), config=config)
        == "unfused-result"
    )
    assert calls == [False]  # upstream's unfused branch was selected
    assert (
        config.apply_rope_fusion is True
    )  # ... and the config is untouched afterwards


def test_fusible_calls_stay_on_the_fused_path(engine, stub_module, monkeypatch):
    calls = []
    module = _dispatcher(engine, stub_module, monkeypatch, calls=calls)
    config = SimpleNamespace(apply_rope_fusion=True, rotary_interleaved=False)

    module.apply_rotary_pos_emb(object(), object(), config=config)
    assert calls == [True]


def test_flag_is_restored_when_the_unfused_call_raises(engine, stub_module):
    _stub_apex(stub_module)

    def dispatch(t, freqs, config, cu_seqlens=None, mscale=1.0, cp_group=None):
        assert config.apply_rope_fusion is False
        raise RuntimeError("boom")

    module = _rope_utils(stub_module, dispatch=dispatch)
    engine.register(_rope.PATCHES)
    engine.install()
    config = SimpleNamespace(apply_rope_fusion=True, rotary_interleaved=True)

    with pytest.raises(RuntimeError):
        module.apply_rotary_pos_emb(object(), object(), config=config)
    assert config.apply_rope_fusion is True


@pytest.mark.parametrize("cp_size,expected", [(1, [True]), (2, [False]), (8, [False])])
def test_packed_sequences_demote_only_under_context_parallel(
    engine, stub_module, monkeypatch, cp_size, expected
):
    calls = []
    module = _dispatcher(engine, stub_module, monkeypatch, calls=calls)
    config = SimpleNamespace(apply_rope_fusion=True, rotary_interleaved=False)
    cp_group = SimpleNamespace(size=lambda: cp_size, rank=lambda: 0)

    module.apply_rotary_pos_emb(
        object(), object(), config=config, cu_seqlens=object(), cp_group=cp_group
    )
    assert calls == expected


def test_context_parallel_lookup_falls_back_to_parallel_state(
    engine, stub_module, monkeypatch
):
    calls = []
    module = _dispatcher(engine, stub_module, monkeypatch, calls=calls)
    cp_group = SimpleNamespace(size=lambda: 4, rank=lambda: 1)
    monkeypatch.setattr(_rope, "_context_parallel_group", lambda: cp_group)
    config = SimpleNamespace(apply_rope_fusion=True, rotary_interleaved=False)

    module.apply_rotary_pos_emb(object(), object(), config=config, cu_seqlens=object())
    assert calls == [False]


def test_unfused_route_passes_through_without_apex_kernels(
    engine, stub_module, monkeypatch
):
    monkeypatch.setattr(_rope, "_apex_kernels", lambda: None)
    calls = []
    module = _dispatcher(engine, stub_module, monkeypatch, calls=calls)
    config = SimpleNamespace(apply_rope_fusion=True, rotary_interleaved=True)

    # The dispatcher wrapper stays installed but leaves upstream's selection
    # intact when there is no apex kernel to adapt.
    assert (
        module.apply_rotary_pos_emb(object(), object(), config=config)
        == "unfused-result"
    )
    assert calls == [True]


# --- hardware round --------------------------------------------------------


@pytest.mark.integration
def test_musa_fused_rope_parity():
    """The installed kernels against a real Megatron and MUSA device.

    Runs in a subprocess: it activates the whole patch set and allocates device
    memory, neither of which belongs in the unit suite's process.
    """
    import os
    import subprocess
    from pathlib import Path

    from tests.conftest import integration_env

    if os.environ.get("TMA_RUN_INTEGRATION") != "1":
        pytest.skip("set TMA_RUN_INTEGRATION=1 with a working Megatron/MUSA stack")
    env = integration_env({"CUDA_DEVICE_MAX_CONNECTIONS": "1"})
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("rope_smoke.py"))],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("ROPE_PASS") == 8, result.stdout


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("supports_interleaved", [False, True])
def test_native_te_interleaved_version_guard(
    engine, stub_module, monkeypatch, packed, supports_interleaved
):
    calls = []
    bshd, thd = Recorder("te-bshd"), Recorder("te-thd")
    extension = stub_module("megatron.core.extensions.transformer_engine")
    extension.fused_apply_rotary_pos_emb = bshd
    extension.fused_apply_rotary_pos_emb_thd = thd
    extension.is_te_min_version = lambda version: supports_interleaved
    module = _dispatcher(
        engine, stub_module, monkeypatch, calls=calls, bshd=bshd, thd=thd
    )
    config = SimpleNamespace(apply_rope_fusion=True, rotary_interleaved=True)
    module.apply_rotary_pos_emb(
        object(), object(), config=config, cu_seqlens=object() if packed else None
    )
    assert calls == [supports_interleaved]
    assert config.apply_rope_fusion is True
    assert module.fused_apply_rotary_pos_emb is bshd


def test_aten_binding_preserves_third_party_kernel(monkeypatch):
    _stub_aten(monkeypatch)
    original = Recorder("third-party")
    original.__module__ = "external_kernels.rope"
    assert _rope._aten_fused_bshd(original) is None


@pytest.mark.parametrize(
    "failure",
    [
        ImportError("broken ABI"),
        OSError("missing shared object"),
        ModuleNotFoundError("internal dependency", name="apex_internal"),
    ],
)
def test_apex_internal_failures_propagate(monkeypatch, failure):
    import builtins

    original_import = builtins.__import__

    def broken(name, *args, **kwargs):
        if name == "fused_rotary_positional_embedding":
            raise failure
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken)
    with pytest.raises(type(failure)) as caught:
        _rope._apex_kernels()
    assert caught.value is failure
