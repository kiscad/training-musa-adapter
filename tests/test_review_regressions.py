"""Regressions for configuration, import safety and transactional failures."""

import importlib
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from training_musa_adaptor import activation
from training_musa_adaptor._config import ConfigManager, Selection
from training_musa_adaptor._engine import AttrPatch
from training_musa_adaptor._errors import (
    ConfigError,
    PatchConflict,
    TrainingMusaAdaptorError,
)

ATTENTION_ID = "megatron.te.attention.capability-dispatch"


def _file(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text)
    return str(path)


def test_env_overrides_file_patch_options(tmp_path, monkeypatch):
    manager = ConfigManager()
    manager.set_config_path(
        _file(
            tmp_path,
            f"""
[patch_options."{ATTENTION_ID}"]
policy = "force"
implementations = ["mate"]
""",
        )
    )
    monkeypatch.setenv("TRAINING_MUSA_ADAPTOR_ATTN_POLICY", "prefer")
    monkeypatch.setenv("TRAINING_MUSA_ADAPTOR_ATTN_IMPLS", "mudnn")
    config = manager.freeze(registered_patch_ids={ATTENTION_ID})
    assert config.options_for(ATTENTION_ID) == Selection(
        "prefer", ("mudnn",), "reference"
    )
    assert config.sources[f'patch_options."{ATTENTION_ID}".policy'] == "env"


@pytest.mark.parametrize(
    "text",
    [
        "patches = 1",
        "attention = 1",
        "patch_options = false",
        f'[patch_options]\n"{ATTENTION_ID}" = 1',
        f'[patches]\nonly = ["{ATTENTION_ID}", "{ATTENTION_ID}"]',
    ],
)
def test_invalid_toml_shapes_are_config_errors(tmp_path, text):
    manager = ConfigManager()
    manager.set_config_path(_file(tmp_path, text))
    with pytest.raises(ConfigError):
        manager.freeze(registered_patch_ids={ATTENTION_ID})
    assert not manager.frozen


def test_failed_config_can_select_a_corrected_file(tmp_path):
    manager = ConfigManager()
    manager.set_config_path(_file(tmp_path, "attention = false"))
    with pytest.raises(ConfigError):
        manager.freeze()
    manager.set_config_path(_file(tmp_path, '[attention]\npolicy = "upstream"'))
    assert manager.freeze().attention().policy == "upstream"


def test_frozen_config_maps_cannot_be_mutated():
    config = ConfigManager().freeze()
    for mapping in (config.operators, config.patch_options, config.sources):
        with pytest.raises(TypeError):
            mapping["x"] = "changed"


def test_explicit_path_after_autoload_is_used(engine, tmp_path, stub_module):
    module = stub_module("config_target", value=1)
    engine.register([AttrPatch("p", "config_target:value", lambda old: 2)])
    engine.install(eager=False)
    engine.install(config_path=_file(tmp_path, '[patches]\ndisable = ["p"]'))
    assert module.value == 1
    assert engine.report()["patches"][0]["status"] == "skipped"
    with pytest.raises(ConfigError, match="frozen"):
        engine.install(config_path="other.toml")


@pytest.mark.parametrize("switch", ["ENABLED", "AUTOLOAD"])
def test_autoload_catches_invalid_switch(switch, monkeypatch):
    monkeypatch.setenv(f"TRAINING_MUSA_ADAPTOR_{switch}", "typo")
    activation.torch_backend_autoload()
    assert activation.bootstrap_errors[-1]["type"] == "ConfigError"


def test_bootstrap_does_not_import_packaging_or_accelerator_libraries():
    script = """
import sys
import training_musa_adaptor as tma
tma.torch_backend_autoload()
assert not tma.bootstrap_errors
for name in ("packaging", "torch", "torchada", "megatron", "transformers"):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        [sys.executable, "-c", script], text=True, capture_output=True
    )
    assert result.returncode == 0, result.stderr


def test_cli_validates_the_same_attention_options(tmp_path, monkeypatch, capsys):
    from training_musa_adaptor.__main__ import main

    path = _file(
        tmp_path,
        f"""
[patch_options."{ATTENTION_ID}"]
policy = "force"
implementations = ["mate"]
""",
    )
    monkeypatch.setenv("TRAINING_MUSA_ADAPTOR_CONFIG", path)
    assert main(["config"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["patch_options"][ATTENTION_ID]["implementations"] == ["mate"]


def test_factory_cannot_overwrite_a_concurrent_writer(engine, stub_module):
    original, external, replacement = object(), object(), object()
    module = stub_module("changed_during_factory", value=original)

    def factory(old):
        module.value = external
        return replacement

    engine.register([AttrPatch("p", "changed_during_factory:value", factory)])
    with pytest.raises(PatchConflict):
        engine.install()
    assert module.value is external
    engine.uninstall()
    assert module.value is external


def test_rollback_failure_retains_cleanup_for_retry(engine, stub_module, monkeypatch):
    from training_musa_adaptor import _engine

    original, replacement = object(), object()
    module = stub_module("rollback_target", value=original)
    real_setattr = setattr

    def writes(owner, name, value):
        if owner is module and value is original:
            raise RuntimeError("restore failed")
        real_setattr(owner, name, value)

    def fail_aliases(*args):
        raise RuntimeError("alias failed")

    engine.register([AttrPatch("p", "rollback_target:value", lambda old: replacement)])
    with monkeypatch.context() as m:
        m.setattr(_engine, "setattr", writes, raising=False)
        m.setattr(_engine, "_rebind_aliases", fail_aliases)
        with pytest.raises(TrainingMusaAdaptorError, match="rollback incomplete"):
            engine.install()
        assert module.value is replacement
        with pytest.raises(PatchConflict, match="cleanup is incomplete"):
            engine.install()
    engine.uninstall()
    assert module.value is original
    assert not engine.report()["cleanup_pending"]


def test_reinstall_keeps_previously_declined_patch_scheduled(engine, fake_package):
    name = fake_package("declined_cycle", "value = 1")
    results = iter((None, 2))
    engine.register([AttrPatch("p", f"{name}:value", lambda old: next(results))])
    engine.install()
    module = importlib.import_module(name)
    assert module.value == 1
    engine.uninstall()
    engine.install()
    assert module.value == 2


@pytest.mark.parametrize("name", ["megatron_musa_patch", "musa_adapter"])
def test_inactive_legacy_import_does_not_block(engine, stub_module, name):
    stub_module(name)
    engine.install()


def test_saved_bootstrap_error_cannot_be_consumed(engine):
    engine._saved_error = RuntimeError("broken bootstrap")
    for _ in range(2):
        with pytest.raises(TrainingMusaAdaptorError, match="broken bootstrap"):
            engine.install()


def test_attention_init_error_does_not_fall_back(monkeypatch):
    from tests.test_attention_ops import meta
    from training_musa_adaptor.ops import attention as ops

    def broken_load():
        raise RuntimeError("broken extension ABI")

    broken = ops.Implementation(
        "mudnn",
        "accelerated",
        lambda m: (True, ""),
        broken_load,
        lambda payload, call: pytest.fail("must not run"),
    )
    monkeypatch.setitem(ops.IMPLEMENTATIONS, "mudnn", broken)
    candidates, reasons = ops.resolve_candidates("auto", (), "reference")
    with pytest.raises(RuntimeError, match="broken extension ABI"):
        ops.select_and_run(
            "test",
            candidates,
            reasons,
            meta(),
            None,
            lambda c: (True, ""),
            lambda c: pytest.fail("must not fallback"),
        )


def test_attention_rejects_before_loading(monkeypatch):
    from tests.test_attention_ops import meta
    from training_musa_adaptor.ops import attention as ops

    impl = ops.Implementation(
        "mate",
        "accelerated",
        lambda m: (False, "unsupported"),
        lambda: pytest.fail("must not load"),
        lambda p, c: None,
    )
    monkeypatch.setitem(ops.IMPLEMENTATIONS, "mate", impl)
    candidates, reasons = ops.resolve_candidates("force", ("mate",), "error")
    with pytest.raises(ops.NoCompatibleImplementation, match="unsupported"):
        ops.select_and_run("test", candidates, reasons, meta(), None, None, None)


@pytest.mark.parametrize("policy", ["force", "upstream"])
def test_attention_policy_applies_before_domain_and_packed_conversion(
    monkeypatch, policy
):
    from training_musa_adaptor.patches.megatron import attention as patch

    selection = Selection(policy, ("mate",) if policy == "force" else (), "error")
    monkeypatch.setattr(patch, "module_source_contains", lambda *a: True)
    monkeypatch.setattr(patch, "_frozen_selection", lambda: (selection, [], []))
    monkeypatch.setattr(patch, "_eligible", lambda *a: policy == "upstream")
    monkeypatch.setattr(patch, "_musa_live", lambda: True)
    monkeypatch.setattr(
        patch,
        "_packed_forward",
        lambda *a: pytest.fail("must not split upstream inputs"),
    )
    calls = []
    original = lambda *a, **kw: calls.append((a, kw)) or "upstream"
    wrapped = patch._tedpa_forward(original)
    tensor = SimpleNamespace(device=SimpleNamespace(type="musa"))
    packed = SimpleNamespace(qkv_format="thd")
    if policy == "force":
        with pytest.raises(Exception, match="outside the supported"):
            wrapped(
                SimpleNamespace(),
                tensor,
                tensor,
                tensor,
                None,
                "causal",
                packed_seq_params=packed,
            )
        assert not calls
    else:
        assert (
            wrapped(
                SimpleNamespace(),
                tensor,
                tensor,
                tensor,
                None,
                "causal",
                packed_seq_params=packed,
            )
            == "upstream"
        )
        assert calls[0][1]["packed_seq_params"] is packed


def test_packed_attention_all_empty_keeps_output_shape_and_gradients():
    import torch

    from training_musa_adaptor.patches.megatron import attention as patch

    q, k, v = [torch.empty(0, 2, 8, requires_grad=True) for _ in range(3)]
    packed = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 0]), cu_seqlens_kv=torch.tensor([0, 0])
    )
    out = patch._packed_forward(SimpleNamespace(), q, k, v, "causal", packed, None)
    assert out.shape == (0, 16)
    out.sum().backward()
    assert all(t.grad is not None for t in (q, k, v))


@pytest.mark.parametrize(
    "module_name,guard,factories",
    [
        (
            "training",
            "musa_available",
            (
                "_enable_pytorch_profile_validate_args",
                "_ignore_overlap_flags_validate_args",
                "_noop_fused_kernels_load",
                "_noop_set_jit_fusion_options",
            ),
        ),
        (
            "layer_norm",
            "_musa_live",
            (
                "_pure_torch_layer_norm",
                "_block_layer_norm_impl",
                "_unfused_te_layer_norm_linear",
            ),
        ),
        ("checkpointing", "musa_available", ("_serial_writer",)),
    ],
)
def test_non_musa_factories_leave_original_behavior(
    module_name, guard, factories, monkeypatch
):
    module = importlib.import_module(
        "training_musa_adaptor.patches.megatron." + module_name
    )
    monkeypatch.setattr(module, guard, lambda: False)
    for name in factories:
        assert getattr(module, name)(object()) is None


def test_softmax_availability_is_probed_once(monkeypatch):
    from training_musa_adaptor.patches.megatron import softmax

    probes = []
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: probes.append(name))
    wrapper = softmax._softmax_kernel_available(
        lambda *a: pytest.fail("missing extension")
    )
    for _ in range(3):
        assert wrapper(None, None, 1, 1, 16, 16) is False
    assert probes == [softmax._SOFTMAX_EXTENSION]
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    assert softmax._softmax_kernel_available(lambda *a: True) is None


def test_attention_report_records_actual_successful_selection(monkeypatch):
    from tests.test_attention_ops import meta
    from training_musa_adaptor.ops import attention as ops

    monkeypatch.setattr(ops, "_LAST_DISPATCH", {})
    implementation = ops.Implementation(
        "mudnn",
        "accelerated",
        lambda m: (True, ""),
        lambda: (True, None),
        lambda payload, call: "result",
    )
    monkeypatch.setitem(ops.IMPLEMENTATIONS, "mudnn", implementation)
    candidates, reasons = ops.resolve_candidates("force", ("mudnn",), "error")
    assert (
        ops.select_and_run("test", candidates, reasons, meta(), None, None, None)
        == "result"
    )
    snapshot = ops.dispatch_report()
    assert snapshot["test"]["implementation"] == "mudnn"
    snapshot["test"]["implementation"] = "changed"
    assert (
        activation.report()["last_attention_dispatch"]["test"]["implementation"]
        == "mudnn"
    )


@pytest.mark.parametrize("backend", ["gdn", "router"])
@pytest.mark.parametrize("missing", ["torch_kernels", "internal_dependency"])
def test_optional_kernel_import_only_declines_missing_package(
    monkeypatch, backend, missing
):
    import builtins

    from training_musa_adaptor.ops import gated_delta_rule
    from training_musa_adaptor.patches.megatron import moe

    original_import = builtins.__import__

    def fail(name, *args, **kwargs):
        if name.startswith("torch_kernels"):
            raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail)
    load = (
        gated_delta_rule._tilelang_stack
        if backend == "gdn"
        else moe._torch_kernels_router_entry_points
    )
    if missing == "torch_kernels":
        assert load() is None
    else:
        with pytest.raises(ModuleNotFoundError, match="internal_dependency"):
            load()
