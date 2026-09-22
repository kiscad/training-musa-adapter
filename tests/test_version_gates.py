# 契约要点: 版本比较使用 packaging SpecifierSet (prereleases=True 显式): rc/dev 参与比较——"2.0rc1 >=2.0" 被拒绝, post/dev 不等于裸 release;
# 语法校验跟随 packaging(~=、==2.0rc1 合法, <2.* 非法); 不存在全局忽略开关;
# module_source_contains 返回 True/False/None(未知), 源码探针不执行目标子模块;
# Hook 门控在 exec 边界评估; 不可解析的安装版本字符串从 check_version_gate
# 抛出 InvalidVersion.
"""Declarative gates must be deterministic and safe before target resolution."""

import importlib
import sys
from types import SimpleNamespace

import pytest
from packaging.version import InvalidVersion

from training_musa_adaptor import _compat
from training_musa_adaptor._engine import AttrPatch, Engine, HookPatch


@pytest.mark.parametrize(
    "installed,bound,blocked",
    [
        ("2.0", "==2.0.0", False),
        ("2.0.0", "<=2.0", False),
        ("2.0.0.1", ">2.0.0", False),
        ("2.0.0+vendor", "==2.0", False),
        # packaging (prereleases=True): an rc is BELOW its release -- the old
        # numeric-release comparison zero-padded (2,0) and let it pass.
        ("2.0rc1", ">=2.0", True),
        # post/dev releases are not the bare release under PEP 440 either
        # (old numeric comparison treated them as equal).
        ("2.0.post1", "==2", True),
        ("2.0.dev1+vendor.1", "==2", True),
    ],
)
def test_release_comparison(monkeypatch, installed, bound, blocked):
    monkeypatch.setattr(_compat, "distribution_version", lambda name: installed)
    assert _compat.check_version_gate("vendor " + bound)[0] is blocked


def test_prerelease_versions_participate_in_comparison(monkeypatch):
    # The task's canonical example: megatron-core 0.16.1rc1 is inside
    # ">=0.14,<0.17" because prereleases are explicitly included.
    monkeypatch.setattr(_compat, "distribution_version", lambda name: "0.16.1rc1")
    assert not _compat.check_version_gate("megatron-core >=0.14,<0.17")[0]
    # An rc below the floor does not get rounded up to it.
    monkeypatch.setattr(_compat, "distribution_version", lambda name: "0.14rc1")
    assert _compat.check_version_gate("megatron-core >=0.14,<0.17")[0]
    # An rc clearly inside the range passes (0.16.9rc3 < 0.17).
    monkeypatch.setattr(_compat, "distribution_version", lambda name: "0.16.9rc3")
    assert not _compat.check_version_gate("megatron-core >=0.14,<0.17")[0]
    # packaging subtlety: "<0.17" excludes even 0.17's own prereleases.
    monkeypatch.setattr(_compat, "distribution_version", lambda name: "0.17rc1")
    assert _compat.check_version_gate("megatron-core >=0.14,<0.17")[0]


@pytest.mark.parametrize("installed", ["unknown", "2..0"])
def test_unparseable_installed_version_raises_loudly(monkeypatch, installed):
    """Invalid installed versions propagate instead of silently skipping."""
    monkeypatch.setattr(_compat, "distribution_version", lambda name: installed)
    with pytest.raises(InvalidVersion):
        _compat.check_version_gate("vendor <2.1")


@pytest.mark.parametrize(
    "spec",
    [
        "vendor >=2junk",
        "vendor <2.*",  # .* is only legal with ==/!= in packaging
        "vendor >=2..0",
        "vendor >>2.0",
        "vendor",  # no operator at all
    ],
)
def test_invalid_gate_syntax_rejected(spec):
    with pytest.raises(ValueError):
        _compat.validate_version_gates((spec,))


@pytest.mark.parametrize(
    "spec",
    [
        "vendor ~=2.0",  # compatible-release operator: valid packaging syntax now
        "vendor ==2.0rc1",  # pinning a prerelease is expressible
        "vendor >=2.0,<2.1",
        "vendor !=2.0",
    ],
)
def test_packaging_gate_syntax_accepted(spec):
    name, _specifier = _compat.parse_version_gate(spec)
    assert name == "vendor"
    _compat.validate_version_gates((spec,))


def test_gate_distribution_names_are_normalized():
    assert (
        _compat.parse_version_gate("Transformer.Engine >=2.0")[0]
        == "transformer-engine"
    )
    assert _compat.parse_version_gate("torch_musa <3")[0] == "torch-musa"


@pytest.mark.parametrize("patch_type", [AttrPatch, HookPatch])
def test_gate_validation_shared(patch_type):
    args = (
        dict(target="example:value", replace=lambda x: x)
        if patch_type is AttrPatch
        else dict(trigger="example", run=lambda: None)
    )
    with pytest.raises(ValueError):
        patch_type("invalid", version_gates=["vendor"], **args)


def test_blocked_gate_does_not_resolve_removed_target(engine, stub_module, monkeypatch):
    monkeypatch.setattr(_compat, "distribution_version", lambda name: "3.0")
    module = stub_module("gate_removed")
    engine.register(
        [
            AttrPatch(
                "removed",
                "gate_removed:OldClass.method",
                lambda x: pytest.fail("must not run"),
                version_gates=("vendor <3",),
            )
        ]
    )
    engine.install()
    assert engine.report()["patches"][0]["status"] == "skipped"
    assert not hasattr(module, "OldClass")


def test_source_probe_handles_top_level_module(tmp_path, monkeypatch):
    source = tmp_path / "probe.py"
    source.write_text("MARKER = True\n")

    def fake_find_spec(name, path=None, target=None):
        if name != "probe":
            return None
        return SimpleNamespace(origin=str(source), submodule_search_locations=None)

    # the probe resolves via find_spec_without_watchers (never executes code)
    import training_musa_adaptor._imports as _imports

    monkeypatch.setattr(_imports, "find_spec_without_watchers", fake_find_spec)
    monkeypatch.setattr(_compat, "find_spec_without_watchers", fake_find_spec)
    assert _compat.module_source_contains("probe", "MARKER") is True
    assert _compat.module_source_contains("probe", "ABSENT") is False
    # contract: True/False/None -- a flat module has no submodule path,
    # so a child probe is a straightforward "absent" (False), not "unknown".
    assert _compat.module_source_contains("probe.child", "MARKER") is False


def test_source_probe_never_executes_the_target_module(
    tmp_path, monkeypatch, tracked_modules
):
    """Source probes resolve paths without executing parent or target bodies."""
    package = tmp_path / "gate_source_probe"
    package.mkdir()
    (package / "__init__.py").write_text(
        "raise RuntimeError('parent must not execute')\n"
    )
    (package / "child.py").write_text(
        "MARKER = 1\nraise RuntimeError('child must not execute')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    tracked_modules.add("gate_source_probe")

    assert _compat.module_source_contains("gate_source_probe.child", "MARKER") is True
    assert "gate_source_probe" not in sys.modules
    assert "gate_source_probe.child" not in sys.modules
    assert _compat.module_source_contains("gate_source_probe.child", "ABSENT") is False
    # a genuinely absent submodule file is False, not "unknown" (None)
    assert (
        _compat.module_source_contains("gate_source_probe.missing", "MARKER") is False
    )


def test_missing_metadata_keeps_target_policy(monkeypatch):
    monkeypatch.setattr(_compat, "distribution_version", lambda name: None)
    assert not _compat.check_version_gate("vendor >=2")[0]


def test_gated_chain_dependencies_and_reinstall(engine, stub_module, monkeypatch):
    monkeypatch.setattr(_compat, "distribution_version", lambda name: "3.0")
    module = stub_module("gate_chain", value=1, consumer=1)
    patches = [
        AttrPatch("consumer", "gate_chain:consumer", lambda old: 9, requires=("old",)),
        AttrPatch(
            "old",
            "gate_chain:value",
            lambda old: old + 10,
            version_gates=("vendor <3",),
        ),
        AttrPatch("always", "gate_chain:value", lambda old: old * 2),
    ]
    engine.register(patches)
    engine.install()
    assert (module.value, module.consumer) == (2, 1)
    assert "requires" in engine.report()["patches"][0]["detail"]
    engine.uninstall()
    assert (module.value, module.consumer) == (1, 1)

    # there is no IGNORE_VERSION_GATES override.  A changed environment
    # is re-evaluated by a fresh engine (config freezes once per engine).
    monkeypatch.setattr(_compat, "distribution_version", lambda name: "2.5")
    second = Engine()
    try:
        second.register(patches)
        second.install()
        assert (module.value, module.consumer) == (22, 9)
        second.uninstall()
        assert (module.value, module.consumer) == (1, 1)
    finally:
        second.uninstall()


def test_hook_gate_rechecked_for_each_engine(engine, fake_package, monkeypatch):
    """Each engine evaluates gates against metadata at its import boundary."""
    calls = []

    def make_hook(trigger):
        return HookPatch(
            "hook",
            trigger,
            lambda: calls.append("run"),
            undo=lambda: calls.append("undo"),
            version_gates=("vendor <3",),
        )

    monkeypatch.setattr(_compat, "distribution_version", lambda name: "3")
    # Both modules are written up front: a file created after the first
    # import would be invisible to the FileFinder's directory cache.
    first_mod = fake_package("gate_hook_first", "")
    second_mod = fake_package("gate_hook_second", "")
    engine.register([make_hook(first_mod)])
    engine.install()
    importlib.import_module(first_mod)
    assert calls == []  # gate blocked at the exec boundary
    assert engine.report()["patches"][0]["status"] == "skipped"
    engine.uninstall()

    monkeypatch.setattr(_compat, "distribution_version", lambda name: "2")
    second = Engine()
    try:
        second.register([make_hook(second_mod)])
        second.install()
        importlib.import_module(second_mod)
        second.uninstall()
        assert calls == ["run", "undo"]
    finally:
        second.uninstall()


def test_metadata_not_cached(monkeypatch):
    versions = iter(["2", "3"])
    monkeypatch.setattr(_compat, "distribution_version", lambda name: next(versions))
    assert not _compat.check_version_gate("vendor <3")[0]
    assert _compat.check_version_gate("vendor <3")[0]


def test_hook_metadata_failure_is_reported(engine, fake_package, monkeypatch):
    def broken_metadata(name):
        raise OSError("metadata unreadable")

    monkeypatch.setattr(_compat, "distribution_version", broken_metadata)
    name = fake_package("gate_error", "")
    engine.register(
        [
            HookPatch(
                "broken",
                name,
                lambda: pytest.fail("must not run"),
                version_gates=("vendor <3",),
            )
        ]
    )
    engine.install()
    with pytest.raises(OSError, match="metadata unreadable"):
        importlib.import_module(name)
    assert engine.report()["patches"][0]["status"] == "failed"
    assert "metadata unreadable" in engine.report()["patches"][0]["detail"]
