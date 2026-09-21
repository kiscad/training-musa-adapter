# 来源: megatron-musa-patch/tests/test_engine_lifecycle.py
# 主要适配点: apply_now() 不存在——attr 目标用 install()+import(或 apply([id]))
# 驱动; Hook 触发模块一律改为未导入的 fake package(已加载模块的 Hook 在 v2.0 标
# phase_missed,不会 late-run); 别名修复显式传 rebind_prefixes=("megatron",)(v2.0
# 默认空); 配置每 Engine 冻结一次,改变选择的用例改用新 Engine 实例;
# 旧"失败版本检查重试"改为冻结失败重试(引擎不再有框架版本检查);
# 旧"多引擎各自应用全部 Hook/属性"按 v2.0 观测语义重写: watcher 互不委托,
# 同一触发模块由最后安装的 watcher 接管边界.
"""Regression coverage for composition, ownership and failure diagnostics."""

from __future__ import annotations

import importlib
import sys

import pytest

from training_musa_adaptor._engine import AttrPatch, Engine, HookPatch
from training_musa_adaptor._errors import (
    PatchConflict,
    PatchTargetMissing,
    TrainingMusaAdaptorError,
)


def _wrap(label):
    def replace(old):
        def wrapper():
            return old() + label

        return wrapper

    return replace


def test_same_target_chain_is_idempotent_and_reloadable(engine, fake_package):
    name = fake_package("chain_target", "def fn(): return 'base'\n")
    engine.register(
        [
            AttrPatch("chain.one", f"{name}:fn", _wrap("1")),
            AttrPatch("chain.two", f"{name}:fn", _wrap("2")),
        ]
    )
    engine.install()
    module = importlib.import_module(name)
    assert module.fn() == "base12"
    engine.install()  # idempotent: no stacked wrappers
    assert module.fn() == "base12"
    importlib.reload(module)
    assert module.fn() == "base12"
    engine.uninstall()
    assert module.fn() == "base"
    engine.install()  # clean reinstall re-applies to the already-imported module
    assert module.fn() == "base12"


def test_register_after_install_rebuilds_chain(engine, stub_module):
    module = stub_module("live_registration", fn=lambda: "base")
    engine.register([AttrPatch("one", "live_registration:fn", _wrap("1"))])
    engine.install()
    engine.register([AttrPatch("two", "live_registration:fn", _wrap("2"))])
    assert module.fn() == "base12"
    engine.uninstall()
    assert module.fn() == "base"


def test_failed_registration_is_atomic(engine):
    patch = AttrPatch("duplicate", "json:loads", lambda old: old)
    with pytest.raises(ValueError):
        engine.register([patch, patch])
    assert engine.report()["patches"] == []


def test_failure_is_recorded_and_target_chain_is_atomic(engine, stub_module):
    module = stub_module("atomic_chain", fn=lambda: "base")
    original = module.fn

    def fail(old):
        raise ValueError("broken factory")

    engine.register(
        [
            AttrPatch("good", "atomic_chain:fn", _wrap("1")),
            AttrPatch("bad", "atomic_chain:fn", fail),
        ]
    )
    with pytest.raises(TrainingMusaAdaptorError, match="broken factory"):
        engine.install()
    assert module.fn is original
    assert engine.report()["patches"][1]["status"] == "failed"
    assert engine.report()["patches"][0]["status"] == "pending"


def test_missing_attribute_is_recorded_failed(engine, stub_module):
    stub_module("missing_attr")
    engine.register([AttrPatch("missing", "missing_attr:nope", lambda old: 1)])
    with pytest.raises(PatchTargetMissing):
        engine.install()
    assert engine.report()["patches"][0]["status"] == "failed"


def test_failed_freeze_can_be_retried(engine, monkeypatch):
    """Ported from the old failed-version-check retry test.  v2.0 has no
    framework version check; the equivalent install-time failure is the
    configuration freeze, and a failed freeze must not brick the engine."""
    state = {"fail": True}
    real_freeze = engine.config.freeze

    def flaky(**kwargs):
        if state["fail"]:
            raise ValueError("config rejected")
        return real_freeze(**kwargs)

    monkeypatch.setattr(engine.config, "freeze", flaky)
    with pytest.raises(ValueError, match="config rejected"):
        engine.install()
    state["fail"] = False
    engine.install()
    assert engine._installed
    assert sys.meta_path.count(engine._watcher) == 1


def test_optional_target_and_broken_dependency_are_distinct(engine, fake_package):
    name = fake_package("broken_dependency", "import missing_dependency_xyz\nvalue = 1\n")
    engine.register([AttrPatch("broken", f"{name}:value", lambda old: 2)])
    with pytest.raises(ModuleNotFoundError, match="missing_dependency_xyz"):
        engine.apply(["broken"])
    assert engine.report()["patches"][0]["status"] == "failed"


def test_internal_import_error_is_not_swallowed(engine, fake_package):
    name = fake_package("broken_import", "raise ImportError('ABI mismatch')\n")
    engine.register([AttrPatch("broken", f"{name}:value", lambda old: 2)])
    with pytest.raises(ImportError, match="ABI mismatch"):
        engine.apply(["broken"])


def test_missing_parent_is_optional(engine):
    engine.register([AttrPatch("optional", "missing_parent_xyz.child:value", lambda old: 2)])
    engine.apply(["optional"])
    assert engine.report()["patches"][0]["status"] == "skipped"


def test_uninstall_clears_pending_and_filter_state(engine, stub_module, monkeypatch):
    module = stub_module("filter_cycle", value=1)
    engine.register([AttrPatch("filter", "filter_cycle:value", lambda old: old + 1)])
    engine.install()
    assert module.value == 2
    engine.uninstall()
    assert module.value == 1
    assert engine.pending_modules() == []

    # v2.0: configuration freezes once per Engine, so a changed selection is
    # observed through a fresh engine (the old test cycled one engine).
    monkeypatch.setenv("TRAINING_MUSA_ADAPTOR_DISABLE", "filter")
    second = Engine()
    try:
        second.register([AttrPatch("filter", "filter_cycle:value", lambda old: old + 1)])
        second.install()
        assert module.value == 1
        assert second.report()["patches"][0]["status"] == "skipped"
    finally:
        second.uninstall()

    monkeypatch.delenv("TRAINING_MUSA_ADAPTOR_DISABLE")
    third = Engine()
    try:
        third.register([AttrPatch("filter", "filter_cycle:value", lambda old: old + 1)])
        third.install()
        assert module.value == 2
    finally:
        third.uninstall()


def test_apply_respects_master_switch_after_install(engine, fake_package, monkeypatch):
    name = fake_package("master_cycle", "value = 1\n")
    engine.register([AttrPatch("master", f"{name}:value", lambda old: 2)])
    engine.install()
    monkeypatch.setenv("TRAINING_MUSA_ADAPTOR_ENABLED", "0")
    engine.apply(["master"])  # hard exit: no import, no application
    assert name not in sys.modules


def test_reversible_and_irreversible_hooks(engine, fake_package):
    name = fake_package("hook_cycle", "")
    calls = []
    engine.register(
        [
            HookPatch(
                "reversible",
                name,
                lambda: calls.append("run"),
                undo=lambda: calls.append("undo"),
            ),
            HookPatch("irreversible", name, lambda: calls.append("once")),
            HookPatch("declined", name, lambda: False),
        ]
    )
    engine.install()
    importlib.import_module(name)
    assert [r["status"] for r in engine.report()["patches"]] == ["applied", "applied", "skipped"]
    engine.uninstall()
    assert calls == ["run", "once", "undo"]
    # A hook without undo is one-shot: reported applied, restart-required.
    assert engine.is_applied("irreversible")
    assert "irreversible" in engine.report()["restart_required"]
    # A clean reinstall may re-run the reversible hook, but only through a
    # real import boundary (never late), so the module is dropped first.
    sys.modules.pop(name)
    engine.install()
    importlib.import_module(name)
    assert calls == ["run", "once", "undo", "run"]


def test_failed_hook_keeps_following_hooks_pending_for_retry(engine, fake_package):
    name = fake_package("hook_retry", "")
    calls = []

    def flaky():
        calls.append("flaky")
        if calls.count("flaky") == 1:
            raise RuntimeError("try again")

    engine.register(
        [
            HookPatch("first", name, lambda: calls.append("first")),
            HookPatch("flaky", name, flaky),
            HookPatch("last", name, lambda: calls.append("last")),
        ]
    )
    engine.install()
    with pytest.raises(RuntimeError, match="try again"):
        importlib.import_module(name)
    # The failed import unwinds; the next import retries the failed hook and
    # runs the ones that were still pending.
    importlib.import_module(name)
    assert calls == ["first", "flaky", "flaky", "last"]


def test_failed_hook_undo_blocks_reinstall_until_unapply_succeeds(engine, fake_package):
    name = fake_package("hook_undo_retry", "")
    state = {"fail_undo": True}

    def undo():
        if state["fail_undo"]:
            raise RuntimeError("undo exploded")
        state["undone"] = True

    engine.register([HookPatch("undo-fail", name, lambda: None, undo=undo)])
    engine.install()
    importlib.import_module(name)
    assert engine.is_applied("undo-fail")

    try:
        with pytest.raises(TrainingMusaAdaptorError, match="undo exploded"):
            engine.uninstall()
        assert engine._records["undo-fail"].status == "failed"
        with pytest.raises(PatchConflict, match="cleanup is incomplete"):
            engine.install()
    finally:
        # A failing undo must not poison the engine fixture's teardown either.
        state["fail_undo"] = False
        engine.uninstall()  # retry succeeds

    assert state["undone"]
    assert not engine.is_applied("undo-fail")
    # Reinstall is allowed again; the already-imported trigger is phase_missed
    # (v2.0 never re-runs a missed hook late).
    engine.install()
    record = engine.report()["patches"][0]
    assert record["status"] == "skipped"
    assert record["detail"].startswith("phase_missed:")


def test_descriptors_and_inherited_attributes_restore_exactly(engine, stub_module):
    class Parent:
        @staticmethod
        def static(value):
            return value + 1

        @classmethod
        def class_method(cls, value):
            return cls.__name__, value

    class Child(Parent):
        pass

    original = vars(Parent)["class_method"]
    stub_module("descriptor_target", Parent=Parent, Child=Child)
    engine.register(
        [
            AttrPatch("static", "descriptor_target:Child.static", lambda old: lambda v: old(v) + 1),
            AttrPatch(
                "class",
                "descriptor_target:Parent.class_method",
                lambda old: lambda cls, v: old(cls, v + 1),
            ),
        ]
    )
    engine.install()
    assert Child().static(1) == 3
    assert Parent.class_method(1) == ("Parent", 2)
    assert Child.class_method(1) == ("Child", 2)
    engine.uninstall()
    assert "static" not in vars(Child)
    assert vars(Parent)["class_method"] is original


def test_alias_repair_is_bounded_and_never_rebinds_flags(engine, stub_module):
    original = lambda: "base"
    target = stub_module("alias_target", fn=original, flag=True)
    inside = stub_module("megatron.alias_test", fn=original, flag=True)
    outside = stub_module("outside_alias", fn=original, flag=True)
    engine.register(
        [
            # v2.0: the old engine repaired megatron.* aliases by default;
            # the scope is now an explicit per-patch declaration.
            AttrPatch("alias", "alias_target:fn", _wrap("1"), rebind_prefixes=("megatron",)),
            AttrPatch("flag", "alias_target:flag", lambda old: False),
        ]
    )
    engine.install()
    assert inside.fn is target.fn
    assert outside.fn is original
    assert inside.flag and outside.flag
    engine.uninstall()
    assert inside.fn is original


def test_uninstall_preserves_external_writes(engine, stub_module):
    target = stub_module("ownership", fn=lambda: "base")
    engine.register([AttrPatch("owned", "ownership:fn", _wrap("1"))])
    engine.install()
    external = lambda: "external"
    target.fn = external
    with pytest.raises(PatchConflict):
        engine.apply(["owned"])
    engine.uninstall()
    assert target.fn is external


def test_multiple_engines_patch_disjoint_modules(engine, fake_package):
    """Two engines coexist: disjoint watched modules each get their engine."""
    name_a = fake_package("engine_a_mod", "value = 1\n")
    name_b = fake_package("engine_b_mod", "value = 1\n")
    other = Engine()
    try:
        engine.register([AttrPatch("a", f"{name_a}:value", lambda old: 10)])
        other.register([AttrPatch("b", f"{name_b}:value", lambda old: 20)])
        engine.install()
        other.install()
        assert importlib.import_module(name_a).value == 10
        assert importlib.import_module(name_b).value == 20
    finally:
        other.uninstall()


def test_multiple_engines_contested_boundary(engine, fake_package):
    """v2.0 watcher semantics on one contested trigger: watchers never
    delegate through each other, so the last-installed watcher wraps the
    loader (the old engine chained find_spec so both engines fired).

    Ported from test_multiple_engines_apply_all_hooks_and_attributes.
    """
    name = fake_package("multiple_engines", "value = 1\n")
    calls = []
    other = Engine()
    try:
        engine.register([HookPatch("first", name, lambda: calls.append(1))])
        other.register(
            [
                HookPatch("second", name, lambda: calls.append(2)),
                AttrPatch("value", f"{name}:value", lambda old: 42),
            ]
        )
        engine.install()
        other.install()
        module = importlib.import_module(name)
        assert calls == [2]  # the last-installed watcher owns this boundary
        assert module.value == 42
        assert not engine.is_applied("first")
        assert engine.pending_modules() == [name]
    finally:
        other.uninstall()
    # With the later engine gone, a fresh import hits the first engine.
    sys.modules.pop(name)
    module = importlib.import_module(name)
    assert calls == [2, 1]
    assert module.value == 1


@pytest.mark.parametrize("target", ["a:b:c", "a..b:c", "a:b.", "a:b..c"])
def test_malformed_targets_fail_early(target):
    with pytest.raises(ValueError, match="invalid"):
        AttrPatch("bad", target, lambda old: old)


@pytest.mark.parametrize("rebuild", [False, True])
def test_alias_commit_failure_restores_target_and_existing_aliases(
    engine, stub_module, monkeypatch, rebuild
):
    from training_musa_adaptor import _engine

    original = lambda: "base"
    module = stub_module("atomic_alias", fn=original)
    alias = stub_module("megatron.atomic_alias", fn=original)
    engine.register(
        [AttrPatch("first", "atomic_alias:fn", _wrap("1"), rebind_prefixes=("megatron",))]
    )
    if rebuild:
        engine.install()
    baseline = module.fn

    def fail(*args):
        raise RuntimeError("alias commit failed")

    with monkeypatch.context() as patcher:
        patcher.setattr(_engine, "_rebind_aliases", fail)
        with pytest.raises(TrainingMusaAdaptorError, match="alias commit failed"):
            if rebuild:
                engine.register(
                    [AttrPatch("second", "atomic_alias:fn", _wrap("2"), rebind_prefixes=("megatron",))]
                )
            else:
                engine.install()
    assert module.fn is baseline
    assert alias.fn is baseline
    engine.uninstall()
    assert module.fn is original
    assert alias.fn is original


def test_failed_attribute_undo_is_retryable_and_does_not_block_other_cleanup(engine, stub_module):
    class Target:
        fail = False

        def __setattr__(self, name, value):
            if self.fail and name == "value" and value == 1:
                raise RuntimeError("attribute undo failed")
            object.__setattr__(self, name, value)

    target = Target()
    target.value = 1
    module = stub_module("retry_undo", target=target, other=1)
    engine.register(
        [
            AttrPatch("other", "retry_undo:other", lambda old: 2),
            AttrPatch("target", "retry_undo:target.value", lambda old: 2),
        ]
    )
    engine.install()
    target.fail = True
    try:
        with pytest.raises(TrainingMusaAdaptorError, match="attribute undo failed"):
            engine.uninstall()
        assert module.other == 1
        assert target.value == 2
        with pytest.raises(PatchConflict, match="cleanup is incomplete"):
            engine.install()
    finally:
        target.fail = False
        engine.uninstall()
    assert target.value == 1


def test_reload_retires_declined_patch_and_repairs_old_consumers(engine, fake_package, stub_module):
    state = {"needed": True}
    name = fake_package("conditional_reload", "def fn(): return 'upstream'\n")
    engine.register(
        [
            AttrPatch(
                "conditional",
                f"{name}:fn",
                lambda old: _wrap("patched")(old) if state["needed"] else None,
                rebind_prefixes=("megatron",),
            )
        ]
    )
    engine.install()
    module = importlib.import_module(name)
    consumer = stub_module("megatron.late_consumer", fn=module.fn)
    state["needed"] = False
    importlib.reload(module)
    assert consumer.fn is module.fn
    assert consumer.fn() == "upstream"
    assert engine.report()["patches"][0]["status"] == "skipped"
    assert not engine._bindings
    engine.uninstall()
    assert consumer.fn is module.fn


@pytest.mark.parametrize("action", ["reload", "register"])
def test_late_aliases_follow_rebuilt_chain(engine, fake_package, stub_module, action):
    name = fake_package("late_alias_target", "def fn(): return 'base'\n")
    engine.register(
        [AttrPatch("first", f"{name}:fn", _wrap("1"), rebind_prefixes=("megatron",))]
    )
    engine.install()
    module = importlib.import_module(name)
    late = stub_module("megatron.late_alias", fn=module.fn)
    foreign = stub_module("megatron.foreign_alias", fn=lambda: "foreign")
    if action == "reload":
        importlib.reload(module)
    else:
        engine.register(
            [AttrPatch("second", f"{name}:fn", _wrap("2"), rebind_prefixes=("megatron",))]
        )
    assert late.fn is module.fn
    assert foreign.fn() == "foreign"
    engine.uninstall()
    assert late.fn is module.fn
    assert late.fn() == "base"
