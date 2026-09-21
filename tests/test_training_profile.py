"""Argument and startup compatibility policies (no GPU required)."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from training_musa_adaptor.patches.megatron import training as _training



@pytest.fixture()
def overlap_policy():
    return _training._ignore_overlap_flags_validate_args


def test_overlap_validator_exception_propagates(overlap_policy):
    original = Mock(side_effect=ValueError("upstream validation failed"))
    with pytest.raises(ValueError, match="upstream validation failed"):
        overlap_policy(original)(SimpleNamespace())


@pytest.mark.parametrize("ckpt_format", ["torch_dist", "torch", "torch_dcp", "fsdp_dtensor"])
def test_live_argument_patch_chain_never_rewrites_checkpoint_format(overlap_policy, ckpt_format):
    args = SimpleNamespace(ckpt_format=ckpt_format, async_save=True, profile=True)
    wrapped = lambda args: args
    for patch in _training.PATCHES:
        if patch.target == "megatron.training.arguments:validate_args":
            replacement = patch.replace(wrapped)
            if replacement is not None:
                wrapped = replacement
    assert wrapped(args) is args
    assert args.ckpt_format == ckpt_format
    assert args.async_save is True
    assert args.use_pytorch_profiler is True
    assert not any(p.id == "megatron.training.ckpt-format.no-torch-dist" for p in _training.PATCHES)


def test_legacy_loader_noop_does_not_call_original():
    original = Mock()
    args = SimpleNamespace(rank=0)
    wrapped = _training._noop_fused_kernels_load(original)
    assert wrapped(args) is None
    assert wrapped.__wrapped__ is original
    original.assert_not_called()


def test_jit_warmup_policy_default_noops(monkeypatch):
    original = Mock()
    assert _training._noop_set_jit_fusion_options(original)() is None
    original.assert_not_called()


