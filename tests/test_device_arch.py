"""Architecture policy and reversible import-triggered hooks, without a GPU."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from training_musa_adaptor.patches.megatron import device_arch as _device_arch
from training_musa_adaptor.patches.megatron import distributed as _distributed


def test_arch_tuple_is_fixed_at_the_verified_default(monkeypatch):
    """The ARCH env knob is retired in any value, any spelling, and
    its absence all yield the verified default for this stack.  A different
    capability claim is a new patch revision, not a deployment knob."""  # legacy name: ignored
    assert _device_arch.arch_tuple() == (8, 3)
    assert _device_arch.arch_major() == 8
    assert _device_arch.arch_tuple() == (8, 3)


def test_arch_wrapper_snapshots_configuration(monkeypatch):
    monkeypatch.setattr(_device_arch, "musa_available", lambda: True)
    original = Mock()
    replacement = _device_arch._replace_arch_version(original)
    assert replacement() == 8  # fixed default; the retired ARCH knob is ignored
    original.assert_not_called()


@pytest.fixture()
def capability_hook(stub_module, monkeypatch):
    monkeypatch.setattr(_device_arch, "musa_available", lambda: True)
    monkeypatch.setattr(_device_arch, "_capability_override", None)
    original = Mock(return_value=(3, 1))
    cuda = SimpleNamespace(get_device_capability=original)
    musa = SimpleNamespace(get_device_capability=original)
    stub_module("torch", cuda=cuda, musa=musa)
    yield cuda, musa, original
    _device_arch._uninstall_torch_capability()


def test_capability_hook_is_reversible_and_idempotent(capability_hook):
    cuda, musa, original = capability_hook
    assert _device_arch._install_torch_capability() is None
    replacement = cuda.get_device_capability
    assert replacement() == (8, 3)
    assert replacement(device=2) == (8, 3)
    assert musa.get_device_capability is original
    assert _device_arch._install_torch_capability() is False
    assert cuda.get_device_capability is replacement
    _device_arch._uninstall_torch_capability()
    assert cuda.get_device_capability is original
    _device_arch._uninstall_torch_capability()
    assert cuda.get_device_capability is original


def test_capability_undo_does_not_overwrite_later_owner(capability_hook):
    cuda, _, _ = capability_hook
    _device_arch._install_torch_capability()
    later = Mock()
    cuda.get_device_capability = later
    _device_arch._uninstall_torch_capability()
    assert cuda.get_device_capability is later


def test_capability_can_be_reinstalled_with_new_configuration(
    capability_hook, monkeypatch
):
    cuda, _, original = capability_hook
    _device_arch._install_torch_capability()
    _device_arch._uninstall_torch_capability()
    _device_arch._install_torch_capability()
    assert cuda.get_device_capability() == (8, 3)
    _device_arch._uninstall_torch_capability()
    assert cuda.get_device_capability is original


def test_capability_undo_restores_uncached_proxy_state(stub_module, monkeypatch):
    native_capability = Mock(return_value=(3, 1))

    class Proxy:
        def __getattr__(self, name):
            if name != "get_device_capability":
                raise AttributeError(name)
            self.get_device_capability = native_capability
            return native_capability

    proxy = Proxy()
    stub_module("torch", cuda=proxy)
    monkeypatch.setattr(_device_arch, "musa_available", lambda: True)
    monkeypatch.setattr(_device_arch, "_capability_override", None)
    try:
        _device_arch._install_torch_capability()
        assert proxy.get_device_capability() == (8, 3)
        _device_arch._uninstall_torch_capability()
        assert "get_device_capability" not in vars(proxy)
        assert proxy.get_device_capability is native_capability
    finally:
        _device_arch._uninstall_torch_capability()


def test_capability_undo_does_not_resolve_deleted_proxy_attribute(
    stub_module, monkeypatch
):
    lookups = []

    class Proxy:
        def __getattr__(self, name):
            lookups.append(name)
            raise AttributeError(name)

    proxy = Proxy()
    stub_module("torch", cuda=proxy)
    monkeypatch.setattr(_device_arch, "musa_available", lambda: True)
    monkeypatch.setattr(_device_arch, "_capability_override", None)
    try:
        _device_arch._install_torch_capability()
        del proxy.get_device_capability
        _device_arch._uninstall_torch_capability()
        assert not lookups
        assert "get_device_capability" not in vars(proxy)
    finally:
        _device_arch._uninstall_torch_capability()


@pytest.fixture()
def teardown_hook(stub_module, monkeypatch):
    monkeypatch.setattr(_distributed, "_musa_live", lambda: True)
    monkeypatch.setattr(_distributed, "_teardown_callback", None)
    dist = stub_module(
        "torch.distributed",
        is_available=Mock(return_value=True),
        is_initialized=Mock(return_value=True),
        destroy_process_group=Mock(),
    )
    stub_module("torch", distributed=dist)
    atexit = stub_module("atexit", register=Mock(), unregister=Mock())
    yield dist, atexit
    _distributed._uninstall_clean_teardown()


def test_teardown_hook_registers_once_and_unregisters_without_destroying(teardown_hook):
    dist, atexit = teardown_hook
    assert _distributed._install_clean_teardown() is True
    callback = atexit.register.call_args.args[0]
    assert _distributed._install_clean_teardown() is False
    atexit.register.assert_called_once_with(callback)
    _distributed._uninstall_clean_teardown()
    atexit.unregister.assert_called_once_with(callback)
    dist.destroy_process_group.assert_not_called()
    assert _distributed._teardown_callback is None
    _distributed._uninstall_clean_teardown()
    atexit.unregister.assert_called_once()


@pytest.mark.parametrize(
    "available,initialized", [(True, True), (True, False), (False, True)]
)
def test_teardown_only_destroys_initialized_available_group(
    teardown_hook, available, initialized
):
    dist, atexit = teardown_hook
    dist.is_available.return_value = available
    dist.is_initialized.return_value = initialized
    _distributed._install_clean_teardown()
    atexit.register.call_args.args[0]()
    assert dist.destroy_process_group.call_count == int(available and initialized)


def test_teardown_failure_is_best_effort(teardown_hook):
    dist, atexit = teardown_hook
    dist.destroy_process_group.side_effect = RuntimeError("already shutting down")
    _distributed._install_clean_teardown()
    atexit.register.call_args.args[0]()
    dist.destroy_process_group.assert_called_once()


def test_import_hooks_register_undo_handlers():
    capability = _device_arch.PATCHES[0]
    teardown = _distributed.PATCHES[0]
    assert capability.undo is _device_arch._uninstall_torch_capability
    assert teardown.undo is _distributed._uninstall_clean_teardown


def test_arch_patches_decline_without_musa(capability_hook, monkeypatch):
    cuda, _, original = capability_hook
    monkeypatch.setattr(_device_arch, "musa_available", lambda: False)
    assert _device_arch._install_torch_capability() is False
    assert cuda.get_device_capability is original
    assert _device_arch._replace_arch_version(original) is None
