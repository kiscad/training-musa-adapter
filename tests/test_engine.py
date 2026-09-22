# 要点: Hook 边界在真实 exec_module——find_spec 查询不触发 Hook、已加载模块的
# Hook 标 phase_missed、namespace trigger 标 failed; ONLY/DISABLE 未知 ID 报
# ConfigError; 点号 target 写法拒绝; Hook 版本门控在 exec 边界评估;
# apply(patch_ids) 必填校验; 已知旧引擎重叠拒绝(EngineOverlapError).
"""Tests for the patch engine itself (no GPU, no framework required)."""

from __future__ import annotations

import importlib
import importlib.util
import sys

import pytest

from training_musa_adaptor._engine import AttrPatch, Engine, HookPatch
from training_musa_adaptor._errors import (
    ConfigError,
    EngineOverlapError,
    PatchTargetMissing,
    TrainingMusaAdaptorError,
)


def test_applies_to_already_imported_module(engine, stub_module):
    module = stub_module("fake_already", value=1)
    engine.register(
        [
            AttrPatch(
                id="t.already",
                target="fake_already:value",
                replace=lambda old: old + 41,
            )
        ]
    )
    engine.install()

    assert module.value == 42
    assert engine.is_applied("t.already")
    assert engine.report()["patches"][0]["status"] == "applied"


def test_applies_on_later_import(engine, fake_package):
    name = fake_package("fake_later", "value = 1\n")
    engine.register(
        [AttrPatch(id="t.later", target=f"{name}:value", replace=lambda old: old + 41)]
    )
    engine.install()

    assert not engine.is_applied("t.later")
    module = importlib.import_module(name)
    assert module.value == 42
    assert engine.is_applied("t.later")


def test_patch_sees_original_and_can_wrap(engine, stub_module):
    def original(x):
        return x * 2

    stub_module("fake_wrap", compute=original)
    seen = {}

    def make_wrapper(old):
        seen["original"] = old

        def wrapper(x):
            return old(x) + 1

        return wrapper

    engine.register(
        [AttrPatch(id="t.wrap", target="fake_wrap:compute", replace=make_wrapper)]
    )
    engine.install()

    assert seen["original"] is original
    assert sys.modules["fake_wrap"].compute(3) == 7


def test_from_import_aliases_are_rebound(engine, fake_package, monkeypatch):
    target = fake_package("fake_def", "def fn():\n    return 'original'\n")
    consumer = fake_package("fake_consumer", f"from {target} import fn\n")

    # Import the consumer first: it captures the *unpatched* function by value,
    # which is exactly the case the engine has to repair.
    consumer_module = importlib.import_module(consumer)
    assert consumer_module.fn() == "original"

    engine.register(
        [
            AttrPatch(
                id="t.rebind",
                target=f"{target}:fn",
                replace=lambda old: (lambda: "patched"),
                # alias repair is an explicit, scoped choice
                # (rebind_prefixes defaults to empty).
                rebind_prefixes=(consumer,),
            )
        ]
    )
    engine.install()

    assert consumer_module.fn() == "patched"


def test_uninstall_restores_original(engine, stub_module):
    module = stub_module("fake_undo", value=1)
    engine.register(
        [AttrPatch(id="t.undo", target="fake_undo:value", replace=lambda old: 99)]
    )
    engine.install()
    assert module.value == 99

    engine.uninstall()
    assert module.value == 1
    assert not engine.is_applied("t.undo")
    assert engine._watcher not in sys.meta_path


def test_reapply_after_patch_is_idempotent(engine, stub_module):
    module = stub_module("fake_idem", value=1)
    engine.register(
        [AttrPatch(id="t.idem", target="fake_idem:value", replace=lambda old: 5)]
    )
    engine.install()
    engine.install()
    engine._apply_for_module("fake_idem", module, trigger="test")

    assert module.value == 5


def test_reload_reapplies_the_patch(engine, fake_package):
    name = fake_package("fake_reload", "value = 1\n")
    module = importlib.import_module(name)

    engine.register(
        [AttrPatch(id="t.reload", target=f"{name}:value", replace=lambda old: 7)]
    )
    engine.install()
    assert module.value == 7

    # ``reload`` re-executes the module body, restoring ``value = 1``; the
    # wrapped loader's after-exec boundary re-applies the patch.
    reloaded = importlib.reload(module)
    assert reloaded.value == 7


def test_nested_target_replaces_attribute_on_owner(engine, stub_module):
    class Holder:
        @staticmethod
        def method():
            return "original"

    stub_module("fake_nested", Holder=Holder)
    engine.register(
        [
            AttrPatch(
                id="t.nested",
                target="fake_nested:Holder.method",
                replace=lambda old: (lambda: "patched"),
            )
        ]
    )
    engine.install()

    assert Holder.method() == "patched"
    engine.uninstall()
    assert Holder.method() == "original"


def test_missing_target_raises_with_context(engine, stub_module):
    stub_module("fake_missing", present=1)
    engine.register(
        [AttrPatch(id="t.missing", target="fake_missing:absent", replace=lambda old: 2)]
    )
    with pytest.raises(PatchTargetMissing) as info:
        engine.install()

    assert "t.missing" in str(info.value)
    # error message quotes the canonical 'module:attr' target form.
    assert "fake_missing:absent" in str(info.value)


def test_absent_module_stays_pending_until_import_attempt(engine):
    engine.register(
        [
            AttrPatch(
                id="t.absent",
                target="definitely_not_installed_xyz:value",
                replace=lambda o: 1,
            )
        ]
    )
    engine.install()
    assert not engine.is_applied("t.absent")
    # The watcher keeps the module until an import attempt resolves it.
    importlib.import_module("json")  # unrelated import must not mark it absent
    assert engine.report()["patches"][0]["status"] == "pending"


def test_declining_patch_is_skipped(engine, stub_module):
    module = stub_module("fake_decline", value=1)
    engine.register(
        [
            AttrPatch(
                id="t.decline", target="fake_decline:value", replace=lambda old: None
            )
        ]
    )
    engine.install()

    assert module.value == 1
    assert engine.report()["patches"][0]["status"] == "skipped"
    assert not engine.is_applied("t.decline")


def test_raising_patch_reports_the_patch_id(engine, stub_module):
    stub_module("fake_boom", value=1)

    def explode(old):
        raise ValueError("nope")

    engine.register([AttrPatch(id="t.boom", target="fake_boom:value", replace=explode)])
    with pytest.raises(TrainingMusaAdaptorError) as info:
        engine.install()

    assert "t.boom" in str(info.value)
    assert "ValueError" in str(info.value)


def test_hook_patch_runs_before_trigger_is_imported(engine, fake_package):
    name = fake_package("fake_trigger", "import sys\nseen = sys.modules['flag']\n")
    calls = []

    engine.register(
        [
            HookPatch(
                id="t.hook",
                trigger=name,
                run=lambda: calls.append("ran") or sys.modules.__setitem__("flag", 1),
            )
        ]
    )
    engine.install()
    assert calls == []

    try:
        importlib.import_module(name)
        assert calls == ["ran"]
        # The hook ran at the before-exec boundary: the module body already
        # sees its effect while executing.
        assert sys.modules[name].seen == 1
    finally:
        sys.modules.pop("flag", None)


def test_find_spec_query_does_not_run_hooks(engine, fake_package):
    """find_spec only *locates and wraps*; an external existence
    query must never fire a hook (including speculative find_spec calls)."""
    name = fake_package("spec_query_target", "value = 1\n")
    calls = []
    engine.register(
        [HookPatch(id="t.query", trigger=name, run=lambda: calls.append(1))]
    )
    engine.install()

    spec = importlib.util.find_spec(name)
    assert spec is not None
    assert calls == []
    assert name not in sys.modules

    importlib.import_module(name)
    assert calls == [1]


def test_hook_on_already_imported_module_is_phase_missed(engine, stub_module):
    """hooks whose boundary was missed are never re-run late."""
    stub_module("already_loaded")
    calls = []
    engine.register(
        [HookPatch(id="t.late", trigger="already_loaded", run=lambda: calls.append(1))]
    )
    engine.install()

    record = engine.report()["patches"][0]
    assert calls == []
    assert record["status"] == "skipped"
    assert record["detail"].startswith("phase_missed:")


def test_hook_on_namespace_package_is_failed(
    engine, tmp_path, monkeypatch, tracked_modules
):
    """a namespace package has no execution body -- it is not a
    valid hook boundary and the hook is marked failed."""
    ns_root = tmp_path / "nsroot"
    (ns_root / "ns_pkg").mkdir(parents=True)  # no __init__.py -> namespace package
    monkeypatch.syspath_prepend(str(ns_root))
    tracked_modules.add("ns_pkg")

    engine.register([HookPatch(id="t.ns", trigger="ns_pkg", run=lambda: None)])
    engine.install()
    importlib.import_module("ns_pkg")

    record = engine.report()["patches"][0]
    assert record["status"] == "failed"
    assert "namespace" in record["detail"]


def test_disable_and_only_filters(engine, stub_module, monkeypatch):
    module = stub_module("fake_filter", a=1, b=1)
    engine.register(
        [
            AttrPatch(id="keep.me", target="fake_filter:a", replace=lambda old: 2),
            AttrPatch(id="drop.me", target="fake_filter:b", replace=lambda old: 2),
        ]
    )
    monkeypatch.setenv("TRAINING_MUSA_ADAPTOR_DISABLE", "drop.me")
    engine.install()

    assert module.a == 2
    assert module.b == 1
    record = {r["id"]: r for r in engine.report()["patches"]}
    assert record["drop.me"]["status"] == "skipped"


def test_only_whitelist_overrides_disable(engine, stub_module, monkeypatch):
    module = stub_module("fake_only", a=1, b=1)
    engine.register(
        [
            AttrPatch(id="want.this", target="fake_only:a", replace=lambda old: 2),
            AttrPatch(id="not.this", target="fake_only:b", replace=lambda old: 2),
        ]
    )
    # A non-empty ONLY is a
    # whitelist and DISABLE is ignored entirely -- even when it names the
    # same id.
    monkeypatch.setenv("TRAINING_MUSA_ADAPTOR_ONLY", "want.this")
    monkeypatch.setenv("TRAINING_MUSA_ADAPTOR_DISABLE", "want.this")
    engine.install()

    assert module.a == 2
    assert module.b == 1
    notes = engine.report()["config"]["notes"]
    assert any("ONLY" in note for note in notes)


def test_master_switch_disables_everything(engine, stub_module, monkeypatch):
    module = stub_module("fake_off", a=1)
    engine.register([AttrPatch(id="t.off", target="fake_off:a", replace=lambda old: 2)])
    monkeypatch.setenv("TRAINING_MUSA_ADAPTOR_ENABLED", "0")
    engine.install()

    assert module.a == 1
    assert engine._watcher not in sys.meta_path


def test_duplicate_patch_id_is_rejected(engine):
    patch = AttrPatch(id="dup", target="json:loads", replace=lambda old: old)
    engine.register([patch])
    with pytest.raises(ValueError, match="duplicate patch id"):
        engine.register([patch])


def test_register_rejects_non_patch_objects(engine):
    with pytest.raises(TypeError, match="AttrPatch or HookPatch"):
        engine.register([lambda old: old])


def test_patch_records_validate_their_fields():
    with pytest.raises(TypeError, match="replace must be callable"):
        AttrPatch(id="t.bad", target="json:loads", replace="nope")
    with pytest.raises(ValueError, match="trigger"):
        HookPatch(id="t.bad", trigger="", run=lambda: None)
    with pytest.raises(TypeError, match="run must be callable"):
        HookPatch(id="t.bad", trigger="megatron", run="nope")


def test_pending_modules_only_lists_unresolved(engine, fake_package):
    name = fake_package("fake_pending", "value = 1\n")
    engine.register(
        [
            AttrPatch(id="t.pending", target=f"{name}:value", replace=lambda old: 2),
            AttrPatch(
                id="t.other", target="never_imported_xyz:value", replace=lambda old: 2
            ),
        ]
    )
    engine.install()
    assert engine.pending_modules() == ["fake_pending", "never_imported_xyz"]

    importlib.import_module(name)
    assert engine.pending_modules() == ["never_imported_xyz"]


def test_multiple_engines_do_not_recurse_on_lookup(engine, fake_package):
    """A private Engine next to another one must not deadlock the import system.

    ``find_spec_without_watchers`` skips every known watcher to prevent
    recursion between engines. The query itself runs no hooks.
    """
    name = fake_package("recurse_target", "value = 1\n")
    calls = []
    package_engine = Engine()  # a second watcher, like the process-wide ENGINE
    try:
        package_engine.register(
            [HookPatch(id="t.package", trigger=name, run=lambda: calls.append(1))]
        )
        package_engine.install()
        engine.register(
            [HookPatch(id="t.private", trigger=name, run=lambda: calls.append(2))]
        )
        engine.install()

        # Any find_spec for the watched module used to raise RecursionError.
        assert importlib.util.find_spec(name) is not None
        assert calls == []
        assert name not in sys.modules
    finally:
        package_engine.uninstall()


def test_rebind_only_touches_identity_matches(engine, stub_module, fake_package):
    """A same-named but different object must not be clobbered."""
    target = fake_package("fake_identity", "value = 1\n")
    other = fake_package("fake_other", "value = 'unrelated'\n")
    engine.register(
        [AttrPatch(id="t.identity", target=f"{target}:value", replace=lambda old: 2)]
    )
    engine.install()

    assert importlib.import_module(other).value == "unrelated"


def test_malformed_target_is_rejected():
    with pytest.raises(ValueError):
        AttrPatch(id="bad", target="no_colon_or_attr", replace=lambda old: old)


def test_unknown_env_ids_are_config_error(engine, stub_module, monkeypatch):
    """Unknown patch ids in ONLY/DISABLE are a hard ConfigError."""
    monkeypatch.setenv("TRAINING_MUSA_ADAPTOR_ONLY", "does-not-exist")
    stub_module("unknown_ids", value=1)
    engine.register(
        [
            AttrPatch(
                id="t.known", target="unknown_ids:value", replace=lambda old: old + 1
            )
        ]
    )
    with pytest.raises(ConfigError, match="unknown patch id"):
        engine.install()


def test_target_must_use_colon_form(engine, stub_module):
    """Targets require module:attr; an ambiguous dotted form is rejected."""
    with pytest.raises(ValueError):
        AttrPatch(id="t.dot", target="dot_form.value", replace=lambda old: old + 1)

    module = stub_module("dot_form", value=1)
    patch = AttrPatch(
        id="t.colon", target="dot_form:value", replace=lambda old: old + 1
    )
    assert patch.module_name == "dot_form" and patch.attr_name == "value"
    engine.register([patch])
    engine.install()
    assert module.value == 2


def test_apply_requires_explicit_known_ids(engine):
    """apply() never mass-imports; ids are required and validated."""
    with pytest.raises(ValueError, match="requires explicit patch_ids"):
        engine.apply(())
    with pytest.raises(ConfigError, match="unknown patch id"):
        engine.apply(("no.such.patch",))


@pytest.mark.parametrize(
    "marker",
    ["__megatron_musa_patch_import_watcher__", "__musa_adapter_import_watcher__"],
)
def test_install_refuses_marked_legacy_finder(engine, monkeypatch, marker):
    """installing next to a known legacy import watcher is refused."""
    finder = type("FakeLegacyFinder", (), {})()
    setattr(finder, marker, True)
    monkeypatch.setattr(sys, "meta_path", [finder] + sys.meta_path)

    with pytest.raises(EngineOverlapError, match="import watcher is already installed"):
        engine.install()
    assert engine._watcher not in sys.meta_path


@pytest.mark.parametrize("module_name", ["megatron_musa_patch", "musa_adapter"])
def test_install_refuses_legacy_module(engine, stub_module, module_name):
    """An already-active external patch package is refused too."""
    stub_module(module_name, is_applied=lambda: True)
    with pytest.raises(EngineOverlapError, match="active patches"):
        engine.install()
    assert engine._watcher not in sys.meta_path


@pytest.fixture
def gate_metadata(monkeypatch):
    from training_musa_adaptor import _compat

    monkeypatch.setattr(_compat, "distribution_version", lambda name: "5.16.1")


def test_version_gate_skips_attr_patch_via_engine(
    engine, monkeypatch, stub_module, gate_metadata
):
    """A blocked declarative gate skips the patch with the reason recorded."""

    original = object()
    module = stub_module("fake.mod", thing=original)
    engine.register(
        [
            AttrPatch(
                id="fake.gated",
                target="fake.mod:thing",
                replace=lambda current: ("replaced",),
                version_gates=("transformers>=999",),
            )
        ]
    )
    engine.install()
    engine._apply_for_module(module.__name__, module, trigger="already-imported")
    record = next(r for r in engine.report()["patches"] if r["id"] == "fake.gated")
    assert record["status"] == "skipped"
    assert "version gate" in record["detail"] and "transformers" in record["detail"]
    # A skipped patch never replaces the binding.
    assert module.thing is original


def test_version_gate_applies_within_range(
    engine, monkeypatch, stub_module, gate_metadata
):
    module = stub_module("fake.mod2", thing=object())
    engine.register(
        [
            AttrPatch(
                id="fake.gated2",
                target="fake.mod2:thing",
                replace=lambda current: ("replaced",),
                version_gates=("transformers>=5",),
            )
        ]
    )
    engine.install()
    engine._apply_for_module(module.__name__, module, trigger="already-imported")
    record = next(r for r in engine.report()["patches"] if r["id"] == "fake.gated2")
    assert record["status"] == "applied", record
    assert sys.modules["fake.mod2"].thing == ("replaced",)


def test_version_gate_skips_hook_via_engine(engine, fake_package, gate_metadata):
    """Hooks honour declarative gates too, recorded as skipped.

    Ported with a not-yet-imported trigger: hooks only run at the real
    exec boundary, and a hook on an already-loaded module is phase_missed --
    it would never reach the gate check.
    """
    name = fake_package("fake_hookmod", "")
    ran = []
    engine.register(
        [
            HookPatch(
                id="fake.hook-gated",
                trigger=name,
                run=lambda: ran.append(1),
                undo=lambda: None,
                version_gates=("transformers>=999",),
            )
        ]
    )
    engine.install()
    importlib.import_module(name)
    assert ran == []
    record = next(r for r in engine.report()["patches"] if r["id"] == "fake.hook-gated")
    assert record["status"] == "skipped"
    assert "version gate" in record["detail"]


def test_version_gate_malformed_spec_rejected():
    with pytest.raises(ValueError):
        AttrPatch(
            id="x",
            target="fake.m:attr",
            replace=lambda c: c,
            version_gates=("transformer_engine >>2.0",),
        )
    with pytest.raises(ValueError):
        AttrPatch(
            id="x",
            target="fake.m:attr",
            replace=lambda c: c,
            version_gates=("transformer_engine",),
        )
    with pytest.raises(ValueError):
        HookPatch(
            id="y",
            trigger="fake.m",
            run=lambda: None,
            version_gates=("transformer_engine <abc",),
        )
