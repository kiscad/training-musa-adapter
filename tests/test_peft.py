# 契约要点: torchao 缺失/版本足够 -> 探针原样返回; 存在但低于 peft 最低版本 ->
# 探针的 ImportError 降级为 False (dispatch_torchao 探针语境的原始语义);
# 非 torchao 的 ImportError 原样透传; 门控 >=0.19,<0.22 (0.14-0.18 最低 0.4.0,
# torchao 0.9.0 通过, 补丁不必要也不应生效); dispatch_torchao 按值绑定探针,
# 只验证 peft.import_utils 边界补丁对已导入 dispatcher 的先后关系用注释说明,
# 不在桩里复刻上游模块结构。
"""peft torchao probe compatibility patch tests."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from training_musa_adaptor import _compat
from training_musa_adaptor.patches import PATCH_SUITES, PATCHES, SUITES
from training_musa_adaptor.patches import peft as _peft


def _wrapped(original):
    patch = next(
        p for p in _peft.PATCHES if p.id == "peft.lora.torchao-probe.version-compat"
    )
    return patch.replace(original)


def test_registration_and_suite():
    ids = [p.id for p in PATCHES]
    assert "peft.lora.torchao-probe.version-compat" in ids
    assert SUITES["peft"] == (_peft,)
    assert PATCH_SUITES["peft.lora.torchao-probe.version-compat"] == "peft"


def test_sufficient_version_keeps_true():
    original = Mock(return_value=True, __name__="is_torchao_available")
    wrapped = _wrapped(original)
    assert wrapped() is True
    original.assert_called_once_with()


def test_missing_torchao_keeps_false():
    original = Mock(return_value=False, __name__="is_torchao_available")
    wrapped = _wrapped(original)
    assert wrapped() is False
    original.assert_called_once_with()


def test_too_old_torchao_import_error_degrades_to_false():
    """The crash this patch exists for: present-but-too-old torchao."""

    def original():
        raise ImportError(
            "Found an incompatible version of torchao. Found version 0.9.0, "
            "but only versions above 0.16.0 are supported"
        )

    original.__name__ = "is_torchao_available"
    assert _wrapped(original)() is False


def test_unrelated_import_error_propagates():
    def original():
        raise ImportError("cannot import name 'x' from 'y'")

    original.__name__ = "is_torchao_available"
    with pytest.raises(ImportError, match="cannot import name"):
        _wrapped(original)()


def test_wrapper_preserves_metadata_and_repeat_calls():
    calls = []

    def original():
        calls.append(1)
        raise ImportError("torchao 0.1.0 is too old")

    original.__name__ = "is_torchao_available"
    wrapped = _wrapped(original)
    assert wrapped.__name__ == "is_torchao_available"
    assert wrapped() is False
    assert wrapped() is False
    assert len(calls) == 2  # the original stays the single source of truth


@pytest.mark.parametrize(
    "installed,applies",
    [
        ("0.19.0", True),
        ("0.19.1", True),
        ("0.20.0", True),
        ("0.21.0", True),
        ("0.18.1", False),  # minimum is 0.4.0 there; torchao 0.9.0 passes
        ("0.14.0", False),
        ("0.22.0", False),  # conservative verification boundary
    ],
)
def test_version_gate_bounds(monkeypatch, installed, applies):
    monkeypatch.setattr(_compat, "distribution_version", lambda name: installed)
    patch = next(
        p for p in _peft.PATCHES if p.id == "peft.lora.torchao-probe.version-compat"
    )
    blocked, _ = _compat.check_version_gate(patch.version_gates[0])
    assert blocked is not applies


def test_engine_applies_patch_at_peft_boundary(stub_module):
    """The AttrPatch lands on peft.import_utils when the module is present."""
    from importlib.metadata import version as dist_version

    original = Mock(return_value=False, __name__="is_torchao_available")
    module = stub_module("peft.import_utils", is_torchao_available=original)

    from training_musa_adaptor._engine import Engine

    engine = Engine()
    engine.register(_peft.PATCHES)
    try:
        engine.install()
        state = {r["id"]: r["status"] for r in engine.report()["patches"]}
        # The real peft (0.19.1) is installed in this environment, so the gate
        # applies; a different installed version would skip instead.
        if dist_version("peft").startswith(("0.19", "0.20", "0.21")):
            assert state["peft.lora.torchao-probe.version-compat"] == "applied"
            assert module.is_torchao_available is not original
            assert module.is_torchao_available() is False
        else:
            assert state["peft.lora.torchao-probe.version-compat"] != "applied"
    finally:
        engine.uninstall()
