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


def test_apex_pair_is_installed_where_upstream_has_none(engine, stub_module):
    apex_bshd, apex_thd = _stub_apex(stub_module)
    module = _rope_utils(stub_module)

    engine.register(_rope.PATCHES)
    engine.install()

    assert [record["status"] for record in engine.report()["patches"]] == ["applied"] * 3
    assert _rope._is_apex_kernel(module.fused_apply_rotary_pos_emb)
    assert _rope._is_apex_kernel(module.fused_apply_rotary_pos_emb_thd)

    sentinel_t, sentinel_freqs = object(), object()
    assert module.fused_apply_rotary_pos_emb(sentinel_t, sentinel_freqs) == "bshd-result"
    assert apex_bshd.calls == [((sentinel_t, sentinel_freqs, False), {})]
    assert (
        module.fused_apply_rotary_pos_emb(sentinel_t, sentinel_freqs, transpose_output_memory=True)
        == "bshd-result"
    )
    assert apex_bshd.calls[-1] == ((sentinel_t, sentinel_freqs, True), {})

    cu_seqlens = object()
    assert (
        module.fused_apply_rotary_pos_emb_thd(sentinel_t, cu_seqlens, sentinel_freqs)
        == "thd-result"
    )
    assert apex_thd.calls == [((sentinel_t, cu_seqlens, sentinel_freqs), {})]


def test_bshd_kernel_refuses_interleaved(engine, stub_module):
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
        module.fused_apply_rotary_pos_emb_thd(object(), object(), object(), cp_size=2, cp_rank=0)


def test_transformer_engine_kernels_are_never_replaced(engine, stub_module):
    _stub_apex(stub_module)
    te_bshd, te_thd = Recorder("te-bshd"), Recorder("te-thd")
    module = _rope_utils(stub_module, bshd=te_bshd, thd=te_thd)

    engine.register(_rope.PATCHES)
    engine.install()

    assert [record["status"] for record in engine.report()["patches"]] == ["skipped", "skipped", "applied"]
    assert module.fused_apply_rotary_pos_emb is te_bshd
    assert module.fused_apply_rotary_pos_emb_thd is te_thd


def test_kernel_patches_decline_when_no_apex_kernels_are_available(
    engine, stub_module, monkeypatch
):
    monkeypatch.setattr(_rope, "_apex_kernels", lambda: None)
    module = _rope_utils(stub_module)

    engine.register(_rope.PATCHES)
    engine.install()

    assert [record["status"] for record in engine.report()["patches"]] == ["skipped", "skipped", "applied"]
    assert module.fused_apply_rotary_pos_emb is None
    assert module.fused_apply_rotary_pos_emb_thd is None


def test_apex_lookup_declines_without_the_compiled_extension(stub_module, monkeypatch):
    bshd, thd = _stub_apex(stub_module)
    assert _rope._apex_kernels() == (bshd, thd)

    monkeypatch.setitem(sys.modules, "fused_rotary_positional_embedding", None)
    assert _rope._apex_kernels() is None



def _dispatcher(engine, stub_module, *, calls, bshd=None, thd=None):
    """Install the fallback trio and return (module, config)."""
    _stub_apex(stub_module)

    def dispatch(t, freqs, config, cu_seqlens=None, mscale=1.0, cp_group=None):
        calls.append(config.apply_rope_fusion)
        return "unfused-result"

    module = _rope_utils(stub_module, bshd=bshd, thd=thd, dispatch=dispatch)
    engine.register(_rope.PATCHES)
    engine.install()
    return module


def test_interleaved_configs_are_demoted_and_the_flag_is_restored(engine, stub_module):
    calls = []
    module = _dispatcher(engine, stub_module, calls=calls)
    config = SimpleNamespace(apply_rope_fusion=True, rotary_interleaved=True)

    assert module.apply_rotary_pos_emb(object(), object(), config=config) == "unfused-result"
    assert calls == [False]  # upstream's unfused branch was selected
    assert config.apply_rope_fusion is True  # ... and the config is untouched afterwards


def test_fusible_calls_stay_on_the_fused_path(engine, stub_module):
    calls = []
    module = _dispatcher(engine, stub_module, calls=calls)
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
    engine, stub_module, cp_size, expected
):
    calls = []
    module = _dispatcher(engine, stub_module, calls=calls)
    config = SimpleNamespace(apply_rope_fusion=True, rotary_interleaved=False)
    cp_group = SimpleNamespace(size=lambda: cp_size, rank=lambda: 0)

    module.apply_rotary_pos_emb(
        object(), object(), config=config, cu_seqlens=object(), cp_group=cp_group
    )
    assert calls == expected


def test_context_parallel_lookup_falls_back_to_parallel_state(engine, stub_module, monkeypatch):
    calls = []
    module = _dispatcher(engine, stub_module, calls=calls)
    cp_group = SimpleNamespace(size=lambda: 4, rank=lambda: 1)
    monkeypatch.setattr(_rope, "_context_parallel_group", lambda: cp_group)
    config = SimpleNamespace(apply_rope_fusion=True, rotary_interleaved=False)

    module.apply_rotary_pos_emb(object(), object(), config=config, cu_seqlens=object())
    assert calls == [False]


def test_unfused_route_passes_through_without_apex_kernels(engine, stub_module, monkeypatch):
    monkeypatch.setattr(_rope, "_apex_kernels", lambda: None)
    calls = []
    module = _dispatcher(engine, stub_module, calls=calls)
    config = SimpleNamespace(apply_rope_fusion=True, rotary_interleaved=True)

    # The dispatcher wrapper stays installed but leaves upstream's selection
    # intact when there is no apex kernel to adapt.
    assert module.apply_rotary_pos_emb(object(), object(), config=config) == "unfused-result"
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
def test_native_te_interleaved_version_guard(engine, stub_module, packed, supports_interleaved):
    calls = []
    bshd, thd = Recorder("te-bshd"), Recorder("te-thd")
    extension = stub_module("megatron.core.extensions.transformer_engine")
    extension.fused_apply_rotary_pos_emb = bshd
    extension.fused_apply_rotary_pos_emb_thd = thd
    extension.is_te_min_version = lambda version: supports_interleaved
    module = _dispatcher(engine, stub_module, calls=calls, bshd=bshd, thd=thd)
    config = SimpleNamespace(apply_rope_fusion=True, rotary_interleaved=True)
    module.apply_rotary_pos_emb(
        object(), object(), config=config, cu_seqlens=object() if packed else None
    )
    assert calls == [supports_interleaved]
    assert config.apply_rope_fusion is True
    assert module.fused_apply_rotary_pos_emb is bshd
