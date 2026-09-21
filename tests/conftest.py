# 来源: megatron-musa-patch/tests/conftest.py
# 主要适配点: 环境隔离覆盖 TRAINING_MUSA_ADAPTOR_* 与旧 MEGATRON_MUSA_PATCH* 前缀;
# 合成 sys.modules 条目集中登记、autouse 清理; 每个测试后就地清理进程级
# activation.ENGINE(不替换对象); 删除 src/ sys.path 注入(包已 editable 安装).
"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
import textwrap
import types

import pytest

# Unit collection must not install the process-wide watcher.  Hardware tests
# activate explicitly in isolated subprocesses.
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

from training_musa_adaptor._engine import Engine  # noqa: E402

#: Environment prefixes every test sees empty.  The legacy prefix is cleaned
#: too: the engine refuses to run next to a known legacy engine, and a stray
#: legacy switch must never change what a test observes.
_ENV_PREFIXES = ("TRAINING_MUSA_ADAPTOR_", "MEGATRON_MUSA_PATCH")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Drop every adaptor/legacy switch before each test.

    monkeypatch restores the pre-test environment afterwards, which also
    removes any switch the test itself set.
    """
    for name in list(os.environ):
        # RUN_INTEGRATION is a test-suite switch, not adaptor configuration.
        if name.startswith(_ENV_PREFIXES) and name != "TMA_RUN_INTEGRATION":
            monkeypatch.delenv(name, raising=False)


#: Names of synthetic modules created via fake_package/stub_module (or
#: registered directly through the ``tracked_modules`` fixture).  The autouse
#: teardown pops them from sys.modules even when a test fails halfway.
_SYNTHETIC_MODULES: set[str] = set()


@pytest.fixture
def tracked_modules():
    """Register extra module names for sys.modules removal after the test."""
    return _SYNTHETIC_MODULES


def _clean_process_engine() -> None:
    """Undo anything a test did to the process-wide activation.ENGINE.

    The engine is cleaned *in place* instead of being replaced:
    ``training_musa_adaptor.__init__`` re-exports activation's functions,
    which close over the activation module-global ENGINE, and a test may
    hold its own reference to it -- rebinding a fresh Engine would leave
    those references pointing at the dirty object.  Clearing every piece of
    mutable state (records, schedules, bindings, undo stack, config freeze,
    registration flag) keeps one ENGINE object with nothing observable left
    over, so state cannot leak into the next test.
    """
    from training_musa_adaptor import activation

    engine = activation.ENGINE
    try:
        engine.uninstall()
    except Exception:
        pass  # teardown must never mask the test's own failure
    engine._records.clear()
    engine._attrs.clear()
    engine._hooks.clear()
    engine._bindings.clear()
    engine._undo_order.clear()
    engine._running_hooks.clear()
    engine._restart_required.clear()
    engine._saved_error = None
    engine._installed = False
    engine._config_ready = False
    engine.config.reset_for_tests()
    activation._registered = False
    activation.bootstrap_errors.clear()


@pytest.fixture(autouse=True)
def _cleanup_global_state():
    """Autouse teardown: synthetic modules, stray watchers, process ENGINE."""
    yield
    for name in list(_SYNTHETIC_MODULES):
        sys.modules.pop(name, None)
    _SYNTHETIC_MODULES.clear()
    # A test's watcher must never survive it (the engine fixture uninstalls
    # its own; this catches engines abandoned mid-failure).
    for finder in list(sys.meta_path):
        if getattr(finder, "__training_musa_adaptor_import_watcher__", False):
            sys.meta_path.remove(finder)
    _clean_process_engine()


@pytest.fixture
def engine(_isolate_env) -> Engine:
    """A private registry, so tests never touch the process-wide one."""
    fresh = Engine()
    yield fresh
    fresh.uninstall()


@pytest.fixture
def fake_package(tmp_path, monkeypatch):
    """Create an importable throw-away package on ``sys.path``.

    Returns a factory ``make(name, source)`` that writes ``<tmp>/<name>.py``
    and returns the dotted module name.  Modules created this way are
    removed from ``sys.modules`` again by the autouse teardown.
    """
    root = tmp_path / "fakepkgs"
    root.mkdir()
    monkeypatch.syspath_prepend(str(root))

    def make(name: str, source: str) -> str:
        path = root / f"{name}.py"
        path.write_text(textwrap.dedent(source))
        _SYNTHETIC_MODULES.add(name)
        return name

    return make


@pytest.fixture
def stub_module(monkeypatch):
    """Install a synthetic module in ``sys.modules`` (cleaned up at teardown)."""

    def make(name: str, **attributes) -> types.ModuleType:
        module = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
        _SYNTHETIC_MODULES.add(name)
        return module

    return make


def integration_env(extra: dict | None = None) -> dict:
    """Environment for subprocess integration tests: adaptor on, importable."""
    from pathlib import Path

    env = {k: v for k, v in os.environ.items() if not k.startswith("TRAINING_MUSA_ADAPTOR")}
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
    paths = [str(Path(__file__).resolve().parents[1] / "src")]
    if env.get("MEGATRON_LM_PATH"):
        paths.append(env["MEGATRON_LM_PATH"])
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    if extra:
        env.update(extra)
    return env
