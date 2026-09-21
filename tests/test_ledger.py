# 来源: megatron-musa-patch/tests/test_ledger.py
# 主要适配点: 维护契约(唯一非空 id、rationale/strategy/upstream/remove_when、
# 显式 MODULES、requires 可解析、补丁模块互不导入)原样保留, 作用于 v2.0 注册表
# (当前 S2: megatron attention 一项); 删除仅属于旧包的检查——megatron 存在性探针
# (megatron_importable/check_megatron_present, 引擎已框架无关)、SUPPORTED_VERSION_SPEC
# /parse_version/_in_supported_range(无全局支持区间, 版本门控见 test_version_gates.py)、
# _env.flag 解析(环境解析集中在 _config); 旧"默认 megatron scope/隐式 rebind"结构断言
# 按 v2.0 重写(显式 rebind_prefixes、trigger 必须为具体模块); 新增 patches/ 顶层
# 重依赖扫描与 find_spec_without_watchers 不唤醒 watcher 的探针测试.
"""Maintenance contract: every patch is identifiable and independently reviewable."""

from __future__ import annotations

import sys

import pytest

from training_musa_adaptor._engine import AppliedPatch, AttrPatch, HookPatch
from training_musa_adaptor.patches import MODULES, PATCHES


def test_patch_ids_are_nonempty_and_unique():
    assert PATCHES
    ids = [patch.id for patch in PATCHES]
    assert all(ids)
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("patch", PATCHES, ids=lambda p: p.id)
def test_every_patch_has_an_actionable_maintenance_record(patch):
    for field in ("rationale", "strategy", "upstream", "remove_when"):
        value = getattr(patch, field)
        assert value.strip(), f"{patch.id} has no {field}"
        if field != "upstream":
            assert len(value.split()) >= 3, f"{patch.id}: {field} must explain, not just label"
        assert AppliedPatch(patch).as_dict()[field] == value


def test_targets_and_hooks_have_explicit_scope():
    """Structural scope invariants (v2.0): the old fixed megatron.* default
    scope and its reviewed-exception table are gone -- alias repair is an
    explicit per-patch declaration and hooks name concrete exec boundaries."""
    for patch in PATCHES:
        if isinstance(patch, AttrPatch):
            assert ":" in patch.target
            assert patch.module_name and patch.attr_name
            # v2.0: rebind_prefixes defaults to empty on purpose; any
            # non-empty scope is a deliberate, reviewed migration choice.
            assert all(isinstance(prefix, str) and prefix for prefix in patch.rebind_prefixes)
        else:
            assert patch.trigger and all(
                part.isidentifier() for part in patch.trigger.split(".")
            ), f"{patch.id}: trigger must name a concrete module boundary"
            # Hooks own their cleanup unless deliberately one-shot.
            assert patch.undo is None or callable(patch.undo)


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_each_module_exports_patch_tuple(module):
    assert isinstance(module.PATCHES, tuple)
    assert module.PATCHES


def test_companion_requirements_are_explicit_and_resolvable():
    from training_musa_adaptor._engine import Engine

    Engine._validate_dependencies(list(PATCHES))
    ids = {patch.id for patch in PATCHES}
    for patch in PATCHES:
        if isinstance(patch, AttrPatch):
            assert set(patch.requires) <= ids
            assert AppliedPatch(patch).as_dict()["requires"] == list(patch.requires)


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_patch_modules_do_not_import_other_patch_modules(module):
    import ast
    from pathlib import Path

    tree = ast.parse(Path(module.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level != 1, f"{module.__name__} imports a sibling patch module"
            assert not (node.module or "").startswith("training_musa_adaptor.patches")
        elif isinstance(node, ast.Import):
            assert not any(
                alias.name.startswith("training_musa_adaptor.patches") for alias in node.names
            )


def test_patch_modules_stay_stdlib_at_top_level():
    """patches/ is imported on the bootstrap path: torch, frameworks and
    accelerator libraries load inside factories, never at module top level."""
    import ast
    from pathlib import Path

    import training_musa_adaptor.patches as patches_pkg

    heavy = {
        "torch",
        "torch_musa",
        "torchada",
        "transformer_engine",
        "megatron",
        "transformers",
        "apex",
        "flash_attn",
        "flash_linear_attention",
        "swift",
        "mcore_bridge",
    }
    root = Path(patches_pkg.__file__).parent
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in tree.body:  # top-level statements only
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                roots = {(node.module or "").split(".")[0]}
            else:
                continue
            offenders = roots & heavy
            assert not offenders, f"{path.name} imports {sorted(offenders)} at top level"


def test_find_spec_probe_does_not_wake_the_import_watcher(engine, fake_package):
    """The internal existence probe skips every import watcher: querying a
    watched module neither fires hooks nor wraps the returned loader
    (replaces the old megatron_importable probe test, which does not exist
    in the framework-agnostic v2.0 engine)."""
    from training_musa_adaptor._imports import find_spec_without_watchers

    name = fake_package("probe_target", "value = 1\n")
    calls = []
    engine.register([HookPatch("t.probe", name, lambda: calls.append(1))])
    engine.install()
    before = set(sys.modules)
    spec = find_spec_without_watchers(name)
    assert spec is not None
    assert calls == []
    assert name not in set(sys.modules) - before
