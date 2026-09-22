"""Gradient timing is part of the delayed-wgrad API, not just numerical parity."""

from collections import deque
from types import SimpleNamespace

import pytest

from training_musa_adaptor.patches.megatron import delayed_wgrad as patch

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("fused", [False, True])
def test_delayed_gradient_timing_and_accumulation(fused):
    owner = SimpleNamespace(_musa_pending_wgrads=deque(), fuse_wgrad_accumulation=fused)
    weight = torch.nn.Parameter(torch.randn(5, 3))
    bias = torch.nn.Parameter(torch.randn(5))
    if fused:
        weight.main_grad = torch.ones_like(weight, dtype=torch.float32)
        weight.grad_added_to_main_grad = False
    expected = torch.ones_like(weight) if fused else torch.zeros_like(weight)
    expected_bias = torch.zeros_like(bias)
    for _ in range(2):
        x = torch.randn(4, 3, requires_grad=True)
        dy = torch.randn(4, 5)
        out = patch._linear(x, weight, bias, owner)
        torch.testing.assert_close(out, torch.nn.functional.linear(x, weight, bias))
        out.backward(dy)
        torch.testing.assert_close(x.grad, dy @ weight)
        expected += dy.T @ x.detach()
        expected_bias += dy.sum(0)
        assert weight.grad is None
    torch.testing.assert_close(bias.grad, expected_bias)
    drain = patch._delayed_backward(lambda self: None)
    drain(owner)
    torch.testing.assert_close(weight.main_grad if fused else weight.grad, expected)
    assert not owner._musa_pending_wgrads
    drain(owner)
    torch.testing.assert_close(weight.main_grad if fused else weight.grad, expected)


def test_delayed_init_preserves_config(monkeypatch):
    monkeypatch.setattr(patch, "musa_available", lambda: True)
    config = SimpleNamespace(delay_wgrad_compute=True, fp8=None)
    owner = SimpleNamespace()

    def init(self, *, config):
        assert not config.delay_wgrad_compute
        self.config, self.tp_size = config, 1

    patch._delayed_init(init)(owner, config=config)
    assert owner.config is config and config.delay_wgrad_compute
    assert not owner._musa_pending_wgrads


def test_delayed_constructor_requires_both_execution_patches():
    from training_musa_adaptor._engine import Engine

    Engine._validate_dependencies(list(patch.PATCHES))
    for p in patch.PATCHES:
        if p.id.endswith("-init"):
            prefix = p.id.removesuffix("init")
            assert set(p.requires) == {prefix + "forward", prefix + "backward"}


@pytest.mark.parametrize("reverse", [False, True])
def test_delayed_patch_dependencies_and_uninstall(
    engine, stub_module, monkeypatch, reverse
):
    monkeypatch.setattr(patch, "musa_available", lambda: True)

    class Linear:
        def __init__(self, *, config):
            if config.delay_wgrad_compute:
                raise RuntimeError("native delay unsupported")
            self.config, self.tp_size = config, 1

        def forward(self, x):
            return x

        def backward_dw(self):
            pass

    originals = (Linear.__init__, Linear.forward, Linear.backward_dw)
    stub_module("megatron.core.extensions.transformer_engine", TELinear=Linear)
    patches = [p for p in patch.PATCHES if ":TELinear." in p.target]
    engine.register(patches[::-1] if reverse else patches)
    engine.install()
    config = SimpleNamespace(delay_wgrad_compute=True, fp8=None)
    module = Linear(config=config)
    assert module.config is config
    assert hasattr(module, "_musa_pending_wgrads")
    engine.uninstall()
    assert (Linear.__init__, Linear.forward, Linear.backward_dw) == originals


@pytest.mark.parametrize("disabled", ["forward", "backward"])
def test_delayed_init_keeps_native_gate_without_execution(
    engine, stub_module, monkeypatch, disabled
):
    monkeypatch.setattr(patch, "musa_available", lambda: True)

    class Linear:
        def __init__(self, *, config):
            if config.delay_wgrad_compute:
                raise RuntimeError("native delay unsupported")

        def forward(self, x):
            return x

        def backward_dw(self):
            pass

    original = Linear.__init__
    stub_module("megatron.core.extensions.transformer_engine", TELinear=Linear)
    patches = [p for p in patch.PATCHES if ":TELinear." in p.target]
    monkeypatch.setenv(
        "TRAINING_MUSA_ADAPTOR_DISABLE", f"megatron.te.telinear.delayed-{disabled}"
    )
    engine.register(patches)
    engine.install()
    assert Linear.__init__ is original
    with pytest.raises(RuntimeError, match="native delay unsupported"):
        Linear(config=SimpleNamespace(delay_wgrad_compute=True))


def test_delayed_wgrad_targets_parameter_after_saved_tensor_unpack():
    owner = SimpleNamespace(_musa_pending_wgrads=deque(), fuse_wgrad_accumulation=False)
    weight = torch.nn.Parameter(torch.randn(5, 3))
    x = torch.randn(4, 3, requires_grad=True)
    dy = torch.randn(4, 5)
    with torch.autograd.graph.saved_tensors_hooks(lambda t: t.detach(), lambda t: t):
        out = patch._linear(x, weight, None, owner)
    out.backward(dy)
    assert weight.grad is None
    patch._delayed_backward(lambda self: None)(owner)
    torch.testing.assert_close(weight.grad, dy.T @ x.detach())


def test_grouped_delayed_return_bias_is_native_list(monkeypatch, stub_module):
    from contextlib import contextmanager

    monkeypatch.setattr(patch, "musa_available", lambda: True)
    stub_module(
        "transformer_engine.pytorch.fp8",
        FP8GlobalStateManager=SimpleNamespace(
            is_fp8_enabled=lambda: False, is_fp8_calibration=lambda: False
        ),
    )

    class Grouped:
        num_gemms = 2
        activation_dtype = torch.float32
        use_bias = True
        te_return_bias = True
        _musa_pending_wgrads = deque()
        weight0 = torch.nn.Parameter(torch.randn(5, 3))
        weight1 = torch.nn.Parameter(torch.randn(5, 3))
        bias0 = torch.nn.Parameter(torch.randn(5))
        bias1 = torch.nn.Parameter(torch.randn(5))

        @contextmanager
        def prepare_forward(self, x, **kwargs):
            yield x

    owner = Grouped()
    x = torch.randn(4, 3, requires_grad=True)
    out, biases = patch._delayed_forward(lambda *args: None)(owner, x, [1, 3])
    assert isinstance(biases, list) and biases == [owner.bias0, owner.bias1]
    expected = torch.cat(
        [
            torch.nn.functional.linear(x[:1], owner.weight0),
            torch.nn.functional.linear(x[1:], owner.weight1),
        ]
    )
    torch.testing.assert_close(out, expected)
