"""Transformer Engine call-shape adapter tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from training_musa_adaptor import _compat
from training_musa_adaptor.patches import transformer_engine as _transformer_engine


def test_te_version_predicate_is_not_overridden(engine, stub_module):
    """The fork's reported version decides; no blanket true result any more."""
    original = Mock(return_value=False, __name__="is_te_min_version")
    module = stub_module("megatron.core.utils", is_te_min_version=original)

    engine.register(_transformer_engine.PATCHES)
    engine.install()

    assert module.is_te_min_version is original
    assert module.is_te_min_version("999.0.0") is False
    original.assert_called_once_with("999.0.0")
    assert all(r["id"] != "megatron.core.utils.te-version-check.ignore" for r in engine.report()["patches"])


def test_mem_monitor_shim_installs_and_undos(monkeypatch):
    monkeypatch.setattr(_transformer_engine, "_te_fork_needs_mem_monitor", lambda: True)
    import sys

    for name in ("musa_patch", "musa_patch.mem_utils"):
        sys.modules.pop(name, None)
    try:
        assert _transformer_engine._install_mem_monitor_shim() is True
        from musa_patch.mem_utils import MemMonitor  # noqa: PLC0415
    finally:
        installed = _transformer_engine._mem_monitor_owned.copy()
        _transformer_engine._uninstall_mem_monitor_shim()
    assert MemMonitor.max_token_num == 0
    # The fork's accounting idiom keeps its behavior.
    MemMonitor.max_token_num = max(MemMonitor.max_token_num, 5)
    assert MemMonitor.max_token_num == 5
    assert not installed.keys() & sys.modules.keys()
    assert not _transformer_engine._mem_monitor_owned


def test_mem_monitor_shim_never_shadows_an_existing_musa_patch(monkeypatch):
    import sys
    import types

    existing = types.ModuleType("musa_patch")
    monkeypatch.setitem(sys.modules, "musa_patch", existing)
    assert _transformer_engine._install_mem_monitor_shim() is False
    assert sys.modules["musa_patch"] is existing
    assert not _transformer_engine._mem_monitor_owned


def test_mem_monitor_shim_requires_the_musa_fork(monkeypatch):
    import sys

    monkeypatch.setattr(_transformer_engine, "_te_fork_needs_mem_monitor", lambda: False)
    assert _transformer_engine._install_mem_monitor_shim() is False
    assert "musa_patch" not in sys.modules


def test_mem_monitor_shim_installs_through_engine(engine, tmp_path, monkeypatch):
    """v2.0: hooks fire at the trigger's real exec boundary, not at install."""
    monkeypatch.setattr(_transformer_engine, "_te_fork_needs_mem_monitor", lambda: True)
    import sys

    sys.modules.pop("musa_patch", None)
    pkg = tmp_path / "megatron" / "core"
    pkg.mkdir(parents=True)
    (tmp_path / "megatron" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "parallel_state.py").write_text("X = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    patch = next(
        p for p in _transformer_engine.PATCHES
        if p.id == "megatron.te.grouped-linear.mem-monitor-compat"
    )
    engine.register([patch])
    try:
        engine.install()
        import megatron.core.parallel_state  # noqa: F401 - fires the boundary
        statuses = {r["id"]: r["status"] for r in engine.report()["patches"]}
        assert statuses["megatron.te.grouped-linear.mem-monitor-compat"] == "applied"
        assert "musa_patch.mem_utils" in sys.modules
    finally:
        engine.uninstall()
        for name in ("megatron.core.parallel_state", "megatron.core", "megatron"):
            sys.modules.pop(name, None)
    assert "musa_patch" not in sys.modules
    assert not _transformer_engine._mem_monitor_owned


class TeFork:
    """Stands in for the MUSA TE fork's ``cpu_offload`` module."""

    def __init__(self, arity=5, variadic=False):
        self.calls = []
        if variadic:

            def target(*args, **kwargs):
                self.calls.append((args, kwargs))
                return "variadic"

        elif arity == 5:

            def target(enabled, num_layers, model_layers, offload_activations, offload_weights):
                self.calls.append(
                    (enabled, num_layers, model_layers, offload_activations, offload_weights)
                )
                return "five"

        else:

            def target(
                enabled,
                num_layers,
                model_layers,
                offload_activations,
                offload_weights,
                double_buffering,
            ):
                self.calls.append(
                    (
                        enabled,
                        num_layers,
                        model_layers,
                        offload_activations,
                        offload_weights,
                        double_buffering,
                    )
                )
                return "six"

        self.target = target


def _cpu_offload_patch():
    return next(
        p
        for p in _transformer_engine.PATCHES
        if p.id == "megatron.te.cpu-offload-context.signature-dispatch"
    )


def _upstream_wrapper(target):
    """Megatron's own six-argument wrapper as it behaves for a TE >= 2.5 report.

    Upstream picks this branch from ``is_te_min_version("2.5.0")``; the adapter
    keeps the call correct when the fork's real signature lags its version.
    """

    def get_cpu_offload_context(
        enabled,
        num_layers,
        model_layers,
        activation_offloading,
        weight_offloading,
        double_buffering,
    ):
        return target(
            enabled,
            num_layers,
            model_layers,
            activation_offloading,
            weight_offloading,
            double_buffering,
        )

    return get_cpu_offload_context


def test_cpu_offload_context_uses_the_fork_signature(engine, stub_module):
    fork = TeFork(arity=5)
    upstream = _upstream_wrapper(fork.target)
    module = stub_module(
        "megatron.core.extensions.transformer_engine",
        _get_cpu_offload_context=fork.target,
        get_cpu_offload_context=upstream,
    )
    engine.register([_cpu_offload_patch()])
    engine.install()

    assert engine.report()["patches"][0]["status"] == "applied"
    assert module.get_cpu_offload_context(True, 4, 4, True, False, True) == "five"
    assert fork.calls == [(True, 4, 4, True, False)]


def test_cpu_offload_context_leaves_a_six_argument_fork_alone(engine, stub_module):
    fork = TeFork(arity=6)
    upstream = _upstream_wrapper(fork.target)
    module = stub_module(
        "megatron.core.extensions.transformer_engine",
        _get_cpu_offload_context=fork.target,
        get_cpu_offload_context=upstream,
    )
    engine.register([_cpu_offload_patch()])
    engine.install()

    assert engine.report()["patches"][0]["status"] == "skipped"
    assert module.get_cpu_offload_context(True, 4, 4, True, False, True) == "six"


def test_cpu_offload_context_declines_on_an_unknown_signature(engine, stub_module):
    fork = TeFork(variadic=True)
    upstream = _upstream_wrapper(fork.target)
    module = stub_module(
        "megatron.core.extensions.transformer_engine",
        _get_cpu_offload_context=fork.target,
        get_cpu_offload_context=upstream,
    )
    engine.register([_cpu_offload_patch()])
    engine.install()

    assert engine.report()["patches"][0]["status"] == "skipped"
    assert module.get_cpu_offload_context is upstream


def test_cpu_offload_context_declines_without_transformer_engine(engine, stub_module):
    module = stub_module(
        "megatron.core.extensions.transformer_engine", get_cpu_offload_context=None
    )
    engine.register([_cpu_offload_patch()])
    engine.install()

    assert engine.report()["patches"][0]["status"] == "skipped"
    assert module.get_cpu_offload_context is None


def test_quantized_init_context_and_ownership(monkeypatch):
    import sys
    import types
    from contextlib import contextmanager

    import pytest

    module = types.ModuleType("transformer_engine.pytorch")
    recipe_module = types.ModuleType("transformer_engine.common.recipe")

    class DelayedScaling:
        pass

    recipe_module.DelayedScaling = DelayedScaling
    active = []

    @contextmanager
    def fp8_model_init(enabled=True, recipe=None, preserve_high_precision_init_val=False):
        active.append((enabled, recipe, preserve_high_precision_init_val))
        try:
            yield
        finally:
            active.pop()

    module.fp8_model_init = fp8_model_init
    import torch

    monkeypatch.setattr(
        torch, "musa", types.SimpleNamespace(is_available=lambda: True), raising=False
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, recipe_module.__name__, recipe_module)
    monkeypatch.setattr(_transformer_engine, "_te_fork_needs_mem_monitor", lambda: True)
    try:
        assert _transformer_engine._install_quantized_model_init()
        assert not _transformer_engine._install_quantized_model_init()
        recipe = DelayedScaling()
        with module.quantized_model_init(recipe=recipe, preserve_high_precision_init_val=True):
            assert active == [(True, recipe, True)]
            with pytest.raises(RuntimeError, match="body failed"):
                with module.quantized_model_init(False):
                    assert len(active) == 2
                    raise RuntimeError("body failed")
            assert len(active) == 1
        assert active == []
        with pytest.raises(NotImplementedError):
            module.quantized_model_init(recipe=object())
        _transformer_engine._uninstall_quantized_model_init()
        assert not hasattr(module, "quantized_model_init")
        assert _transformer_engine._install_quantized_model_init()
        foreign = object()
        module.quantized_model_init = foreign
        _transformer_engine._uninstall_quantized_model_init()
        assert module.quantized_model_init is foreign
        assert not _transformer_engine._install_quantized_model_init()
    finally:
        _transformer_engine._uninstall_quantized_model_init()


def test_jit_script_compat_round_trip(monkeypatch):
    """Scripting resolves the builtin while eager shims stay installed."""
    import torch

    calls = []
    # The ATen builtin, independent of whether the vendor shim currently
    # owns torch.arange (suite order may import transformer_engine first).
    builtin = torch._C._VariableFunctions.arange

    def make_wrapper():
        # Vendor pattern: the builtin is captured under ``original_<attr>``.
        original_arange = builtin

        def patched_arange(*args, **kwargs):
            calls.append(args)
            return original_arange(*args, **kwargs)

        return patched_arange

    def scripted_probe(n: int):
        return torch.arange(n) + 1

    wrapper = make_wrapper()
    wrapper.__module__ = "transformer_engine.musa"
    previous = torch.arange
    monkeypatch.setattr(torch, "arange", wrapper)
    script_before = torch.jit.script
    try:
        assert _transformer_engine._install_jit_script_compat() is True
        assert torch.jit.script is not script_before
        assert torch.jit.script(scripted_probe)(3).tolist() == [1, 2, 3]
        # The scripted graph called the builtin; the eager wrapper stayed installed.
        assert calls == []
        assert torch.arange is wrapper

        assert _transformer_engine._install_jit_script_compat() is False
        _transformer_engine._uninstall_jit_script_compat()
        assert torch.jit.script is script_before
        # The vendor wrapper owns eager calls again after undo.
        assert torch.arange is wrapper
        assert torch.arange(3).tolist() == [0, 1, 2]
        assert calls[-1] == (3,)
    finally:
        _transformer_engine._uninstall_jit_script_compat()
        torch.arange = previous


def test_factory_shim_compat_end_to_end_subprocess():
    """With auto-activation, eager torch.jit.script over factories survives TE."""
    pytest.importorskip("torch_musa")
    if not __import__("torch").musa.is_available():
        pytest.skip("no visible MUSA device")
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ,
        TORCH_DEVICE_BACKEND_AUTOLOAD="1",
        TRAINING_MUSA_ADAPTOR_ENABLED="1",
        TRAINING_MUSA_ADAPTOR_AUTOLOAD="1",
        OMP_NUM_THREADS="1",
    )
    code = (
        "import torch, transformer_engine\n"
        "from training_musa_adaptor import report\n"
        "def scripted_probe(n: int, device: torch.device):\n"
        "    # ms-swift's zigzag_ring_attn.get_half_lse shape: factory calls\n"
        "    # inside an eager @torch.jit.script function.\n"
        "    return torch.arange(n, device=device) + torch.empty(1, device=device)\n"
        "ref = torch.randn(8, device='musa')\n"
        "torch.jit.script(scripted_probe)(4, ref.device)\n"
        "x = torch.empty(4, device='cuda')\n"
        "assert x.device.type == 'musa', x.device\n"
        "statuses = {r['id']: r['status'] for r in report()['patches']}\n"
        "assert statuses.get('megatron.te.factory-shim.torchscript-compat') == 'applied', statuses\n"
        "print('FACTORY_SHIM_OK')\n"
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "factory_shim_probe.py"
        probe.write_text(code, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(probe)],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(root),
        )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FACTORY_SHIM_OK" in result.stdout


def test_jit_uninstall_preserves_foreign_wrapper(monkeypatch):
    import functools

    import torch

    before = torch.jit.script
    try:
        assert _transformer_engine._install_jit_script_compat()
        owned = torch.jit.script

        @functools.wraps(owned)
        def foreign(*args, **kwargs):
            return owned(*args, **kwargs)

        monkeypatch.setattr(torch.jit, "script", foreign)
        _transformer_engine._uninstall_jit_script_compat()
        assert torch.jit.script is foreign
    finally:
        _transformer_engine._uninstall_jit_script_compat()
        torch.jit.script = before


def test_jit_compile_keeps_eager_factory_binding(monkeypatch):
    """An eager caller must never observe an untranslated factory during compile."""
    import torch

    original_arange = torch._C._VariableFunctions.arange

    def patched_arange(*args, **kwargs):
        return original_arange(*args, **kwargs)

    patched_arange.__module__ = "transformer_engine.musa"
    monkeypatch.setattr(torch, "arange", patched_arange)

    from torch.jit import _builtins

    table = _builtins._get_builtin_table()
    table.pop(id(patched_arange), None)

    def compiler(fn, *args, **kwargs):
        assert table[id(patched_arange)] == "aten::arange"
        assert torch.arange is patched_arange
        raise ValueError("compile error")

    monkeypatch.setattr(torch.jit, "script", compiler)
    try:
        assert _transformer_engine._install_jit_script_compat()
        with pytest.raises(ValueError, match="compile error"):
            torch.jit.script(lambda: None)
        assert torch.arange is patched_arange
        assert id(patched_arange) not in table
    finally:
        _transformer_engine._uninstall_jit_script_compat()


@pytest.mark.parametrize(
    "install",
    [
        _transformer_engine._install_jit_script_compat,
        _transformer_engine._install_safe_te_utils_module,
    ],
)
def test_early_te_hooks_decline_non_musa_fork(monkeypatch, install):
    monkeypatch.setattr(_transformer_engine, "_te_fork_needs_mem_monitor", lambda: False)
    try:
        assert install() is False
    finally:
        _transformer_engine._uninstall_jit_script_compat()
        _transformer_engine._uninstall_safe_te_utils_module()


@pytest.mark.parametrize("foreign", [False, True])
def test_safe_utils_lifecycle_preserves_module_ownership(monkeypatch, foreign):
    import sys
    import types

    name = "transformer_engine.musa.pytorch.utils"
    parent = types.ModuleType("transformer_engine.musa.pytorch")
    monkeypatch.setitem(sys.modules, parent.__name__, parent)
    monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(_transformer_engine, "_utils_module_owned", None)
    monkeypatch.setattr(_transformer_engine, "_te_fork_needs_mem_monitor", lambda: True)
    try:
        assert _transformer_engine._install_safe_te_utils_module()
        owned = sys.modules[name]
        parent.utils = owned
        assert owned.__package__ == parent.__name__
        assert owned.__spec__.name == name
        original, replacement = object(), object()
        target = types.SimpleNamespace(value=original)
        owned.replace_attr(target, "value", replacement)
        assert target.value is replacement and target._orig_value is original
        assert not _transformer_engine._install_safe_te_utils_module()
        if foreign:
            other = types.ModuleType(name)
            sys.modules[name] = parent.utils = other
        _transformer_engine._uninstall_safe_te_utils_module()
        if foreign:
            assert sys.modules[name] is other and parent.utils is other
        else:
            assert name not in sys.modules and not hasattr(parent, "utils")
    finally:
        _transformer_engine._uninstall_safe_te_utils_module()


def test_factory_shim_skips_when_vendor_shim_removed(monkeypatch):
    """A TE build without factory wrappers no longer owns torch.jit.script."""
    import torch

    script_before = torch.jit.script
    monkeypatch.setattr(_compat, "module_source_contains", lambda name, *m: False)
    assert _transformer_engine._install_jit_script_compat() is False
    assert torch.jit.script is script_before


def test_factory_shim_installs_when_shim_marker_present(monkeypatch):
    import torch

    monkeypatch.setattr(_compat, "module_source_contains", lambda name, *m: True)
    script_before = torch.jit.script
    try:
        assert _transformer_engine._install_jit_script_compat() is True
        assert torch.jit.script is not script_before
    finally:
        _transformer_engine._uninstall_jit_script_compat()
    assert torch.jit.script is script_before


def test_safe_seed_skips_when_vendor_loop_fixed(monkeypatch):
    """A repaired vendor utils module must not be shadowed by the replica."""
    import sys

    monkeypatch.setattr(_compat, "module_source_contains", lambda name, *m: False)
    name = "transformer_engine.musa.pytorch.utils"
    monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(_transformer_engine, "_utils_module_owned", None)
    monkeypatch.setattr(_transformer_engine, "_te_fork_needs_mem_monitor", lambda: True)
    assert _transformer_engine._install_safe_te_utils_module() is False
    assert name not in sys.modules


def test_safe_seed_seeds_when_unsafe_loop_present(monkeypatch):
    import sys

    name = "transformer_engine.musa.pytorch.utils"
    saved = sys.modules.pop(name, None)
    monkeypatch.setattr(_transformer_engine, "_utils_module_owned", None, raising=False)
    monkeypatch.setattr(_compat, "module_source_contains", lambda name_, *m: True)
    try:
        assert _transformer_engine._install_safe_te_utils_module() is True
        assert name in sys.modules
    finally:
        _transformer_engine._uninstall_safe_te_utils_module()
        if saved is not None:
            sys.modules[name] = saved


def test_mem_monitor_shim_respects_unimported_package(monkeypatch, fake_package):
    import sys

    fake_package("musa_patch", "raise RuntimeError('probe must not execute package')\n")
    monkeypatch.delitem(sys.modules, "musa_patch", raising=False)
    monkeypatch.delitem(sys.modules, "musa_patch.mem_utils", raising=False)
    monkeypatch.setattr(_transformer_engine, "_te_fork_needs_mem_monitor", lambda: True)
    try:
        assert _transformer_engine._install_mem_monitor_shim() is False
        assert "musa_patch" not in sys.modules
        assert not _transformer_engine._mem_monitor_owned
    finally:
        _transformer_engine._uninstall_mem_monitor_shim()
