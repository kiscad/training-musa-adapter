"""The teardown hook owns only its callback, independently of device adaptation."""

import atexit
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from training_musa_adaptor.patches.megatron import distributed as _distributed


@pytest.mark.parametrize("initialized", [False, True])
def test_teardown_is_independent_and_uninstall_does_not_destroy_groups(
    engine, stub_module, monkeypatch, initialized, tmp_path, tracked_modules
):
    """v2.0: the teardown hook fires at its trigger's real exec boundary."""
    callbacks = []
    dist = SimpleNamespace(
        is_available=lambda: True, is_initialized=lambda: initialized, destroy_process_group=Mock()
    )
    pkg = tmp_path / "megatron" / "core"
    pkg.mkdir(parents=True)
    (tmp_path / "megatron" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "parallel_state.py").write_text("X = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    tracked_modules.update(("megatron", "megatron.core", "megatron.core.parallel_state"))
    stub_module("torch", distributed=dist)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    monkeypatch.setattr(atexit, "register", callbacks.append)
    monkeypatch.setattr(atexit, "unregister", callbacks.remove)
    patch = next(p for p in _distributed.PATCHES if p.id == "torch.distributed.clean-teardown")
    engine.register([patch])
    engine.install()
    engine.install()
    import megatron.core.parallel_state  # noqa: F401 - fires the boundary
    assert len(callbacks) == 1
    callbacks[0]()
    assert dist.destroy_process_group.call_count == int(initialized)
    engine.uninstall()
    assert callbacks == []
    assert dist.destroy_process_group.call_count == int(initialized)
    assert _distributed._teardown_callback is None


def _premul_patch():
    return next(
        p for p in _distributed.PATCHES if p.id == "megatron.fsdp.premul-sum.device-prescale"
    )


def test_fsdp_premul_sum_branch_prescales_and_uses_sum(stub_module):
    """The PREMUL_SUM branch becomes an in-place device prescale + SUM."""
    calls = []

    class Buffer:
        dtype = float  # not bfloat16

        def mul_(self, factor):
            calls.append(factor)

    recorded = {}

    def original(grad_data, scaling_factor, ddp_config):
        recorded["called"] = True
        return "ORIGINAL_OP"

    reduce_op = SimpleNamespace(SUM="SUM")
    stub_module("torch", bfloat16="bf16-marker", distributed=SimpleNamespace(ReduceOp=reduce_op))
    stub_module("torch.distributed", ReduceOp=reduce_op)

    patch = _premul_patch()
    wrapped = patch.replace(original)
    config = SimpleNamespace(average_in_collective=False, gradient_reduce_div_fusion=True)
    buffer = Buffer()
    assert wrapped(buffer, 0.5, config) == "SUM"
    assert calls == [0.5]
    assert "called" not in recorded
    # bf16 keeps the upstream prescale-in-else branch, other branches delegate.
    buffer.dtype = "bf16-marker"
    assert wrapped(buffer, 0.5, config) == "ORIGINAL_OP"
    assert calls == [0.5]  # the original branch owns its own scaling
    config.average_in_collective = True
    buffer.dtype = float
    assert wrapped(buffer, 0.5, config) == "ORIGINAL_OP"
    config.average_in_collective = False
    config.gradient_reduce_div_fusion = False
    assert wrapped(buffer, 0.5, config) == "ORIGINAL_OP"
    assert wrapped(buffer, None, config) == "ORIGINAL_OP"
    assert calls == [0.5]


def test_fsdp_premul_patch_registered_target():
    patch = _premul_patch()
    assert patch.target.endswith("param_and_grad_buffer:gradient_reduce_preprocessing")
    assert "PREMUL_SUM" in patch.rationale
    assert "PreMulSum" in patch.strategy or "SUM" in patch.strategy


def _subgroups_patch():
    return next(
        p for p in _distributed.PATCHES if p.id == "megatron.bridge-communicator.subgroups-backend"
    )


def _subgroups_env(musa_available=True):
    import types

    recorded = {}

    class Dist:
        def __getattr__(self, name):
            raise AttributeError(name)

        def new_subgroups_by_enumeration(self, *args, **kwargs):
            recorded["args"], recorded["kwargs"] = args, kwargs
            return "current", ["subgroups"]

    types.SimpleNamespace(musa=types.SimpleNamespace(is_available=lambda: musa_available))
    import torch  # the runtime musa probe reads the real namespace

    monkey_musa = types.SimpleNamespace(is_available=lambda: musa_available)
    original_musa = getattr(torch, "musa", None)
    torch.musa = monkey_musa
    proxy = _subgroups_patch().replace(Dist())
    return proxy, recorded, torch, original_musa


def test_subgroups_keyword_nccl_is_translated():
    proxy, seen, torch, original_musa = _subgroups_env()
    try:
        out = proxy.new_subgroups_by_enumeration([[0, 1]], backend="nccl", group_desc="bridge")
        assert out == ("current", ["subgroups"])
        assert seen["kwargs"]["backend"] == "mccl"
        assert seen["kwargs"]["group_desc"] == "bridge"
    finally:
        if original_musa is not None:
            torch.musa = original_musa


def test_subgroups_positional_nccl_is_translated():
    proxy, seen, torch, original_musa = _subgroups_env()
    try:
        proxy.new_subgroups_by_enumeration([[0, 1]], None, "nccl")
        assert seen["args"] == ([[0, 1]], None, "mccl")
        assert "backend" not in seen["kwargs"]
    finally:
        if original_musa is not None:
            torch.musa = original_musa


def test_subgroups_backend_enum_and_passthrough():
    proxy, seen, torch, original_musa = _subgroups_env()

    class Backend(str):
        NCCL = "nccl"

    try:
        proxy.new_subgroups_by_enumeration([[0, 1]], backend=Backend.NCCL)
        assert seen["kwargs"]["backend"] == "mccl"
        proxy.new_subgroups_by_enumeration([[0, 1]], backend="gloo")
        assert seen["kwargs"]["backend"] == "gloo"
        proxy.new_subgroups_by_enumeration([[0, 1]], backend="mccl")
        assert seen["kwargs"]["backend"] == "mccl"
        proxy.new_subgroups_by_enumeration([[0, 1]], backend=None)
        assert seen["kwargs"]["backend"] is None
        proxy.new_subgroups_by_enumeration([[0, 1]])
        assert "backend" not in seen["kwargs"]
    finally:
        if original_musa is not None:
            torch.musa = original_musa


def test_subgroups_untouched_without_musa_runtime():
    proxy, seen, torch, original_musa = _subgroups_env(musa_available=False)
    try:
        proxy.new_subgroups_by_enumeration([[0, 1]], backend="nccl")
        assert seen["kwargs"]["backend"] == "nccl"
    finally:
        if original_musa is not None:
            torch.musa = original_musa


def test_subgroups_proxy_forwards_everything_else():
    import types

    dist = types.SimpleNamespace(get_rank=lambda: 3, is_initialized=lambda: True)
    proxy = _subgroups_patch().replace(dist)
    assert proxy.get_rank() == 3
    assert proxy.is_initialized() is True
    with pytest.raises(AttributeError):
        proxy.nonexistent  # noqa: B018 -- deliberate missing-attribute probe


def test_subgroups_patches_target_the_megatron_callers():
    ids = {p.id: p for p in _distributed.PATCHES}
    bridge = ids["megatron.bridge-communicator.subgroups-backend"]
    grid = ids["megatron.hyper-comm-grid.subgroups-backend"]
    assert bridge.target == "megatron.core.pipeline_parallel.bridge_communicator:dist"
    assert grid.target == "megatron.core.hyper_comm_grid:dist"
    assert bridge.rebind_prefixes == ("megatron",)
    assert grid.rebind_prefixes == ("megatron",)
