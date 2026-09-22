# 要点: `import training_musa_adaptor` 不自动 install——显式 install()/uninstall()
# 才装/卸 watcher; 自动通道只有 torch.backends entry point, kill switch 为
# TRAINING_MUSA_ADAPTOR_AUTOLOAD; eager=False 只装 watcher、配置延迟到首个
# 相关边界冻结; 禁用的激活完全不触碰引擎; 子进程不注入 PYTHONPATH(包已安装).
"""Public activation and entry-point loading, isolated from unit registries."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest


def _run(script, **switches):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("MEGATRON_MUSA_PATCH")
        and not k.startswith("TRAINING_MUSA_ADAPTOR_")
    }
    env.update(switches)
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_explicit_import_never_installs_anything():
    _run("""
        import sys
        import training_musa_adaptor as m
        assert 'torch' not in sys.modules
        assert 'megatron' not in sys.modules
        from training_musa_adaptor import activation
        # importing the package alone installs no watcher.
        assert not activation.ENGINE._installed
        assert not any(
            getattr(f, '__training_musa_adaptor_import_watcher__', False)
            for f in sys.meta_path
        )
        m.install()
        assert activation.ENGINE._installed
        m.uninstall()
        assert not activation.ENGINE._installed
        assert not any(
            getattr(f, '__training_musa_adaptor_import_watcher__', False)
            for f in sys.meta_path
        )
    """)


def test_autoload_entrypoint_installs_only_the_watcher():
    _run("""
        import training_musa_adaptor as m
        from training_musa_adaptor import activation
        m.torch_backend_autoload()
        assert activation.ENGINE._installed
        # eager=False: config parsing is deferred to the first relevant
        # import boundary so `import torch` stays light.
        assert not activation.ENGINE._config_ready
        assert not activation.bootstrap_errors
        m.uninstall()
    """)


def test_entrypoint_load_honours_autoload_switch():
    _run(
        """
        import training_musa_adaptor as m
        from training_musa_adaptor import activation
        m.torch_backend_autoload()
        assert not activation.ENGINE._installed
        m.install()  # explicit activation still works with the channel off
        assert activation.ENGINE._installed
        m.uninstall()
    """,
        TRAINING_MUSA_ADAPTOR_AUTOLOAD="0",
    )


def test_entrypoint_catches_install_failure():
    _run("""
        import training_musa_adaptor as m
        from training_musa_adaptor import activation
        def fail(**kwargs): raise RuntimeError('test failure')
        activation.ENGINE.install = fail
        m.torch_backend_autoload()  # must not break `import torch`
        assert not activation.ENGINE._installed
        assert activation.bootstrap_errors
        assert activation.bootstrap_errors[0]['type'] == 'RuntimeError'
        assert 'test failure' in activation.bootstrap_errors[0]['message']
    """)


def test_disabled_activation_never_touches_the_engine(monkeypatch):
    """TRAINING_MUSA_ADAPTOR_ENABLED=0 is a hard exit: install()/apply()
    return before registering, scheduling or touching the watcher."""
    from training_musa_adaptor import activation

    monkeypatch.setenv("TRAINING_MUSA_ADAPTOR_ENABLED", "0")

    def forbidden(*args, **kwargs):
        pytest.fail("disabled activation touched the engine")

    monkeypatch.setattr(activation.ENGINE, "install", forbidden)
    monkeypatch.setattr(activation.ENGINE, "apply", forbidden)
    activation.install()
    activation.apply(("megatron.te.attention.capability-dispatch",))
    assert not activation.ENGINE._installed
