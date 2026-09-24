# 契约要点: get_norm_dtype 在无 fp64 norm kernel 的设备上返回 fp32 (MUSA 加入
# DeepSpeed 自身的 MPS 先例); 非 MUSA (CUDA/CPU) 保持 torch.double; torch 在工厂内
# 导入; 门控 >=0.19,<0.20 (get_norm_dtype 0.19.0 引入, 0.17.2/0.18.9 无此函数);
# stage3/stage_1_and_2 按值绑定该名字, 补丁在 zero.utils exec 边界先行生效。
"""DeepSpeed ZeRO grad-norm dtype patch tests."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from training_musa_adaptor import _compat
from training_musa_adaptor.patches import PATCH_SUITES, SUITES
from training_musa_adaptor.patches import deepspeed as _deepspeed


def _wrapped(original):
    patch = next(
        p for p in _deepspeed.PATCHES if p.id == "deepspeed.zero.grad-norm.fp32"
    )
    return patch.replace(original)


def test_registration_and_suite():
    assert any(p.id == "deepspeed.zero.grad-norm.fp32" for p in _deepspeed.PATCHES)
    assert SUITES["deepspeed"] == (_deepspeed,)
    assert PATCH_SUITES["deepspeed.zero.grad-norm.fp32"] == "deepspeed"


def test_cuda_double_kept(monkeypatch):
    """On CUDA (no torch.musa) the original fp64 selection is untouched."""
    monkeypatch.delattr(torch, "musa", raising=False)
    original = Mock(return_value=torch.double, __name__="get_norm_dtype")
    assert _wrapped(original)() is torch.double
    original.assert_called_once_with()


def test_musa_double_becomes_float(monkeypatch):
    """The MUSA case: the dtype exists but the norm kernel does not."""
    monkeypatch.setattr(
        torch,
        "musa",
        type("M", (), {"is_available": staticmethod(lambda: True)})(),
        raising=False,
    )
    original = Mock(return_value=torch.double, __name__="get_norm_dtype")
    assert _wrapped(original)() is torch.float


def test_musa_float32_passthrough(monkeypatch):
    """A future DeepSpeed that already returns fp32 is not changed again."""
    monkeypatch.setattr(
        torch,
        "musa",
        type("M", (), {"is_available": staticmethod(lambda: True)})(),
        raising=False,
    )
    original = Mock(return_value=torch.float, __name__="get_norm_dtype")
    assert _wrapped(original)() is torch.float


def test_metadata_and_boundary_documented():
    patch = next(
        p for p in _deepspeed.PATCHES if p.id == "deepspeed.zero.grad-norm.fp32"
    )
    assert "stage3" in patch.strategy or "stage_1_and_2" in patch.strategy
    assert "torch" not in patch.strategy or "factory" in patch.strategy


@pytest.mark.parametrize(
    "installed,applies",
    [
        ("0.19.0", True),
        ("0.19.7", True),
        ("0.18.9", False),  # get_norm_dtype does not exist there
        ("0.17.2", False),
        ("0.20.0", False),  # conservative verification boundary
    ],
)
def test_version_gate_bounds(monkeypatch, installed, applies):
    monkeypatch.setattr(_compat, "distribution_version", lambda name: installed)
    patch = next(
        p for p in _deepspeed.PATCHES if p.id == "deepspeed.zero.grad-norm.fp32"
    )
    blocked, _ = _compat.check_version_gate(patch.version_gates[0])
    assert blocked is not applies
