"""Grouped GEMM reference: numerics, autograd and vendor-precedence."""

import pytest

from training_musa_adaptor.patches.megatron import grouped_gemm as _grouped_gemm

torch = pytest.importorskip("torch")


def test_gmm_matches_per_expert_matmul_reference():
    generator = torch.Generator().manual_seed(3)
    counts = [2, 0, 3]
    total = sum(counts)
    a = torch.randn(total, 5, generator=generator)
    b = torch.randn(3, 5, 4, generator=generator)
    out = _grouped_gemm.gmm(a, b, torch.tensor(counts), trans_b=False)
    assert out.shape == (total, 4)
    start = 0
    for count, weight in zip(counts, b):
        expected = a[start : start + count] @ weight
        torch.testing.assert_close(out[start : start + count], expected)
        start += count


def test_gmm_trans_b():
    a = torch.randn(4, 3)
    b = torch.randn(2, 5, 3)  # stored as [E, N, K]; trans_b uses b^T per expert
    counts = [2, 2]
    out = _grouped_gemm.gmm(a, b, torch.tensor(counts), trans_b=True)
    assert out.shape == (4, 5)
    start = 0
    for count, weight in zip(counts, b):
        expected = a[start : start + count] @ weight.transpose(-2, -1)
        torch.testing.assert_close(out[start : start + count], expected)
        start += count


def test_gmm_rejects_inconsistent_counts():
    a = torch.randn(4, 3)
    with pytest.raises(ValueError, match="tokens_per_expert sums"):
        _grouped_gemm.gmm(a, torch.randn(2, 3, 4), torch.tensor([1, 1]))


def test_gmm_keeps_autograd_and_empty_expert_gradients():
    a = torch.randn(5, 4, requires_grad=True)
    b = torch.randn(2, 4, 3, requires_grad=True)
    counts = torch.tensor([5, 0])
    out = _grouped_gemm.gmm(a, b, counts)
    out.sum().backward()
    assert a.grad is not None and a.grad.shape == a.shape
    # The empty expert still owns a zero gradient instead of losing the edge.
    assert b.grad is not None
    assert torch.equal(b.grad[1], torch.zeros_like(b.grad[1]))
    assert torch.count_nonzero(b.grad[0]) > 0


def test_ops_patch_declines_when_vendor_exists(monkeypatch):
    import sys

    module = types_module()
    monkeypatch.setitem(sys.modules, _grouped_gemm._UTIL, module)
    monkeypatch.setitem(module.__dict__, "grouped_gemm", object())
    assert _grouped_gemm._grouped_gemm_ops(object()) is None
    assert _grouped_gemm._grouped_gemm_is_available(object()) is None
    assert _grouped_gemm._assert_grouped_gemm_is_available(object()) is None


def types_module():
    import sys
    import types

    name = "megatron.core.transformer.moe.grouped_gemm_util"
    module = sys.modules.get(name) or types.ModuleType(name)
    return module


def test_ops_patch_installs_when_vendor_missing(monkeypatch):
    import sys
    import types

    name = _grouped_gemm._UTIL
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(module, "grouped_gemm", None, raising=False)

    ops = _grouped_gemm._grouped_gemm_ops(None)
    assert isinstance(ops, _grouped_gemm._GroupedGemmOps)
    available = _grouped_gemm._grouped_gemm_is_available(lambda: False)
    assert available() is True
    assert_fn = _grouped_gemm._assert_grouped_gemm_is_available(lambda: None)
    assert_fn()  # patched flag says available: no assertion


def test_assert_reports_when_fallback_off(monkeypatch):
    import sys
    import types

    name = _grouped_gemm._UTIL
    module = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(module.__dict__, "grouped_gemm", None)
    monkeypatch.setitem(module.__dict__, "grouped_gemm_is_available", lambda: False)
    assert_fn = _grouped_gemm._assert_grouped_gemm_is_available(lambda: None)
    with pytest.raises(AssertionError, match="Grouped GEMM is not available"):
        assert_fn()


def test_grouped_gemm_patches_are_registered():
    ids = {p.id for p in _grouped_gemm.PATCHES}
    assert {
        "megatron.moe.grouped-gemm.torch-ops",
        "megatron.moe.grouped-gemm.available-flag",
        "megatron.moe.grouped-gemm.assert-noop",
    } <= ids
    for patch in _grouped_gemm.PATCHES:
        if patch.id in {
            "megatron.moe.grouped-gemm.available-flag",
            "megatron.moe.grouped-gemm.assert-noop",
        }:
            assert patch.requires == ("megatron.moe.grouped-gemm.torch-ops",)


@pytest.mark.parametrize("counts", [[2, 2, 0], [4], [-1, 5]])
def test_gmm_rejects_invalid_expert_partitions(counts):
    with pytest.raises(ValueError, match="tokens_per_expert"):
        _grouped_gemm.gmm(torch.randn(4, 3), torch.randn(2, 3, 5), torch.tensor(counts))


def test_gmm_zero_experts_preserves_gradient_edges():
    a = torch.empty(0, 3, requires_grad=True)
    b = torch.empty(0, 3, 5, requires_grad=True)
    output = _grouped_gemm.gmm(a, b, torch.empty(0, dtype=torch.long))
    assert output.shape == (0, 5)
    output.sum().backward()
    assert a.grad is not None
    assert b.grad is not None
