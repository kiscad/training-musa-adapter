"""CPU contracts for the functional LayerNorm/RMSNorm fallback."""

from types import SimpleNamespace

import pytest

from training_musa_adaptor.patches.megatron import layer_norm as _layer_norm


@pytest.fixture(autouse=True)
def _musa_factory_environment(monkeypatch):
    # CPU contract tests exercise the replacement independently of host hardware.
    monkeypatch.setattr(_layer_norm, "_musa_live", lambda: True)


torch = pytest.importorskip("torch")


def _config(normalization="LayerNorm", zero_centered=False, sequence_parallel=True):
    return SimpleNamespace(
        normalization=normalization,
        layernorm_zero_centered_gamma=zero_centered,
        sequence_parallel=sequence_parallel,
        persist_layer_norm=True,
    )


@pytest.fixture()
def norm_class():
    return _layer_norm._pure_torch_layer_norm(object())


@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("hidden_size", [4, (2, 4), torch.Size([4])])
def test_forward_backward_matches_reference(
    norm_class, normalization, zero_centered, hidden_size
):
    config = _config(normalization, zero_centered)
    # Deliberately omit the normalization keyword, exactly as TransformerBlock does.
    norm = norm_class(config, hidden_size, eps=1e-5).double()
    shape = norm.hidden_size
    assert isinstance(shape, torch.Size)
    x = torch.randn((3, *shape), dtype=torch.float64, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_(True)
    with torch.no_grad():
        norm.weight.uniform_(-0.25, 0.25)
        if norm.bias is not None:
            norm.bias.uniform_(-0.25, 0.25)
    reference_weight = norm.weight.detach().clone().requires_grad_(True)
    reference_bias = (
        norm.bias.detach().clone().requires_grad_(True)
        if norm.bias is not None
        else None
    )
    gamma = reference_weight + 1 if zero_centered else reference_weight
    if normalization == "LayerNorm":
        reference = torch.nn.functional.layer_norm(
            reference_x, shape, gamma, reference_bias, 1e-5
        )
    else:
        dimensions = tuple(range(-len(shape), 0))
        reference = reference_x * torch.rsqrt(
            reference_x.square().mean(dimensions, keepdim=True) + 1e-5
        )
        reference = reference * gamma
    output = norm(x)
    torch.testing.assert_close(output, reference)
    assert (
        output._base is None
    ), "pipeline output deallocation requires a viewless tensor"
    gradient = torch.randn_like(output)
    output.backward(gradient)
    reference.backward(gradient)
    torch.testing.assert_close(x.grad, reference_x.grad)
    torch.testing.assert_close(norm.weight.grad, reference_weight.grad)
    if reference_bias is not None:
        torch.testing.assert_close(norm.bias.grad, reference_bias.grad)


@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
@pytest.mark.parametrize("zero_centered", [False, True])
def test_initialization_state_dict_and_optimizer_markers(
    norm_class, normalization, zero_centered
):
    config = _config(normalization, zero_centered)
    # Config wins over the legacy constructor hints, like upstream's implementation.
    norm = norm_class(config, 4, zero_centered_gamma=not zero_centered)
    torch.testing.assert_close(
        norm.weight, torch.full((4,), 0.0 if zero_centered else 1.0)
    )
    assert norm.weight.sequence_parallel is True
    assert norm.sequence_parallel is True
    assert norm.persist_layer_norm is False
    assert norm.config is config
    if normalization == "RMSNorm":
        assert norm.bias is None
        assert set(norm.state_dict()) == {"weight"}
    else:
        torch.testing.assert_close(norm.bias, torch.zeros(4))
        assert norm.bias.sequence_parallel is True
        assert set(norm.state_dict()) == {"weight", "bias"}
    clone = norm_class(config, 4)
    clone.load_state_dict(norm.state_dict(), strict=True)
    x = torch.randn(2, 4)
    torch.testing.assert_close(norm(x), clone(x))
    with torch.no_grad():
        norm.weight.fill_(9)
    norm.reset_parameters()
    torch.testing.assert_close(norm.weight, clone.weight)


def test_unknown_normalization_fails_instead_of_silently_using_layernorm(norm_class):
    with pytest.raises(ValueError, match="Unsupported normalization"):
        norm_class(_config("TypoNorm"), 4)


def test_config_normalization_overrides_constructor_default(norm_class):
    norm = norm_class(_config("RMSNorm"), 4, normalization="LayerNorm")
    assert norm.normalization == "RMSNorm"
    assert norm.bias is None


def test_constructor_normalization_is_fallback_for_older_config(norm_class):
    config = _config()
    del config.normalization
    norm = norm_class(config, 4, normalization="RMSNorm")
    assert norm.normalization == "RMSNorm"
    assert norm.bias is None


def test_block_norm_uses_current_local_class(stub_module, monkeypatch, norm_class):
    monkeypatch.delenv("MEGATRON_MUSA_PATCH_BLOCK_LAYERNORM", raising=False)
    stub_module("megatron.core.fusions.fused_layer_norm", FusedLayerNorm=norm_class)
    assert _layer_norm._block_layer_norm_impl(object()) is norm_class


@pytest.mark.parametrize("installed", [False, True])
def test_availability_flags_require_installed_fallback(
    stub_module, norm_class, installed
):
    local_class = norm_class if installed else type("UpstreamNorm", (), {})
    stub_module("megatron.core.fusions.fused_layer_norm", FusedLayerNorm=local_class)
    assert _layer_norm._fallback_available_flag(False) is (True if installed else None)
    assert _layer_norm._persistent_available_flag(True) is (
        False if installed else None
    )


@pytest.mark.parametrize(
    ("patch_id", "expected"),
    [
        (
            "megatron.te.layer-norm-linear.unfused",
            ("megatron-core >=0.9,<0.20", "transformer_engine >=2.0,<2.1"),
        ),
        (
            "megatron.transformer-block.layer-norm.impl-local",
            ("megatron-core >=0.8,<0.20", "transformer_engine >=2.0,<2.1"),
        ),
        (
            "megatron.te.norm.unfused-musa",
            ("megatron-core >=0.9,<0.20", "transformer_engine >=2.0,<2.1"),
        ),
    ],
)
def test_layer_norm_te_patches_declare_version_gates(patch_id, expected):
    """TE-abort fallbacks carry the megatron-core envelope of their Megatron
    target alongside the declarative 2.0.x TE gate."""
    gates = {p.id: p.version_gates for p in _layer_norm.PATCHES}
    assert gates[patch_id] == expected
