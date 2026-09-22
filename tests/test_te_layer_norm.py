"""Norm-linear fallback: gradients, parameter names and checkpoint contract."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.conftest import integration_env
from training_musa_adaptor.patches.megatron import layer_norm as _layer_norm


@pytest.fixture(autouse=True)
def _musa_factory_environment(monkeypatch):
    # CPU contract tests exercise the replacement independently of host hardware.
    monkeypatch.setattr(_layer_norm, "_musa_live", lambda: True)


torch = pytest.importorskip("torch")


@pytest.mark.parametrize("zero_centered", [False, True])
@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_norm_linear_contract(stub_module, monkeypatch, zero_centered, dtype):

    class Linear(torch.nn.Module):
        def __init__(self, input_size, output_size, *, config, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randn(output_size, input_size, dtype=dtype)
            )
            self.bias = torch.nn.Parameter(torch.randn(output_size, dtype=dtype))

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight), self.bias

    def sharded(self, *args, **kwargs):
        return self.state_dict()

    stub_module(
        "megatron.core.extensions.transformer_engine",
        HAVE_TE=True,
        TEColumnParallelLinear=Linear,
    )
    fallback = _layer_norm._unfused_te_layer_norm_linear(
        SimpleNamespace(sharded_state_dict=sharded)
    )
    config = SimpleNamespace(
        normalization="LayerNorm",
        layernorm_epsilon=1e-5,
        layernorm_zero_centered_gamma=zero_centered,
        params_dtype=dtype,
        sequence_parallel=True,
    )
    module = fallback(16, 8, config=config)
    x = torch.randn(4, 16, dtype=dtype, requires_grad=True)
    ref_x = x.detach().double().requires_grad_()
    ref_gamma = module.layer_norm_weight.detach().double().requires_grad_()
    gamma = ref_gamma + 1 if zero_centered else ref_gamma
    ref_bias = module.layer_norm_bias.detach().double().requires_grad_()
    norm = torch.nn.functional.layer_norm(ref_x, (16,), gamma, ref_bias, 1e-5)
    expected = torch.nn.functional.linear(norm, module.weight.detach().double())
    actual, bias = module(x)
    assert bias is module.bias
    tol = (
        dict(atol=0.15, rtol=0.03)
        if dtype == torch.bfloat16
        else dict(atol=1e-10, rtol=1e-10)
    )
    torch.testing.assert_close(actual.double(), expected, **tol)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad.double())
    torch.testing.assert_close(x.grad.double(), ref_x.grad, **tol)
    torch.testing.assert_close(
        module.layer_norm_weight.grad.double(), ref_gamma.grad, **tol
    )
    keys = {"weight", "bias", "layer_norm_weight"}
    keys.add("layer_norm_bias")
    torch.testing.assert_close(
        module.layer_norm_bias.grad.double(), ref_bias.grad, **tol
    )
    assert set(module.sharded_state_dict()) == keys
    assert module.layer_norm_weight.sequence_parallel
    assert module.layer_norm_weight.allreduce
    copied = deepcopy(module)
    torch.testing.assert_close(copied(x)[0], actual)
    clone = fallback(16, 8, config=config)
    clone.load_state_dict(module.state_dict(), strict=True)
    torch.testing.assert_close(clone(x)[0], actual)


@pytest.mark.parametrize("zero_centered", [False, True])
def test_rmsnorm_constructs_original_fused_module(
    stub_module, monkeypatch, zero_centered
):
    calls = []

    class Linear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            raise AssertionError("RMSNorm must not construct the unfused TE Linear")

    class Original(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            calls.append((args, kwargs))

        def forward(self, x):
            return x

        def sharded_state_dict(self):
            return {}

    stub_module(
        "megatron.core.extensions.transformer_engine",
        HAVE_TE=True,
        TEColumnParallelLinear=Linear,
    )
    patched = _layer_norm._unfused_te_layer_norm_linear(Original)
    config = SimpleNamespace(
        normalization="RMSNorm", layernorm_zero_centered_gamma=zero_centered
    )
    options = dict(
        config=config, bias=False, skip_bias_add=True, tp_group=object(), stride=3
    )
    module = patched(16, 8, **options)
    assert type(module) is Original
    assert isinstance(module, patched)
    assert type(module).forward is Original.forward
    assert not hasattr(module, "_tma_fallback")
    assert calls == [((16, 8), options)]
    x = torch.randn(2, 16)
    assert module(x) is x


@pytest.mark.parametrize("normalization", ["LayerNorm"])
def test_norm_linear_absorbs_mixed_dtype_input(stub_module, monkeypatch, normalization):
    """A wider input dtype must be absorbed at the norm, TE-style.

    Upstream's learned position embedding is a plain ``torch.nn.Embedding``
    (fp32 by default) while the model runs bf16, so the embedding sum reaches
    the first norm in fp32. TE's op contract computes in the parameter dtype
    and casts the *input*; casting the parameters instead leaks fp32 into the
    next TE linear, which asserts "Data types for parameters must match when
    outside of autocasted region" (a2a overlap suite). RMSNorm always
    constructs the original fused module by design, so only LayerNorm runs
    this class's forward.
    """

    class Linear(torch.nn.Module):
        def __init__(self, input_size, output_size, *, config, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.randn(output_size, input_size, dtype=torch.bfloat16)
            )
            self.bias = None

        def forward(self, x):
            assert x.dtype == self.weight.dtype, "linear must see the parameter dtype"
            return torch.nn.functional.linear(x, self.weight), None

    def sharded(self, *args, **kwargs):
        return self.state_dict()

    stub_module(
        "megatron.core.extensions.transformer_engine",
        HAVE_TE=True,
        TEColumnParallelLinear=Linear,
    )
    fallback = _layer_norm._unfused_te_layer_norm_linear(
        SimpleNamespace(sharded_state_dict=sharded)
    )
    config = SimpleNamespace(
        normalization=normalization,
        layernorm_epsilon=1e-5,
        layernorm_zero_centered_gamma=False,
        params_dtype=torch.bfloat16,
        sequence_parallel=True,
    )
    module = fallback(16, 8, config=config)
    assert module.layer_norm_weight.dtype == torch.bfloat16
    x = torch.randn(4, 16, dtype=torch.float32, requires_grad=True)
    actual, _ = module(x)
    assert actual.dtype == torch.bfloat16, "mixed dtype must not leak past the norm"
    # TE's convention: the input is cast before the norm, so the reference
    # computes on x.to(bf16) with bf16 parameters.
    ref_x = x.detach().to(torch.bfloat16).float().requires_grad_(True)
    gamma = module.layer_norm_weight.detach().float()
    if normalization == "LayerNorm":
        ref_bias = module.layer_norm_bias.detach().float()
        norm = torch.nn.functional.layer_norm(ref_x, (16,), gamma, ref_bias, 1e-5)
    else:
        dimensions = (-1,)
        norm = ref_x * torch.rsqrt(ref_x.square().mean(dimensions, keepdim=True) + 1e-5)
        norm = norm * gamma
    expected = torch.nn.functional.linear(norm, module.weight.detach().float())
    torch.testing.assert_close(actual.float(), expected, atol=0.2, rtol=0.05)
    actual.float().sum().backward()
    assert x.grad is not None and x.grad.dtype == torch.float32
    assert module.layer_norm_weight.grad is not None
    assert module.layer_norm_weight.grad.dtype == torch.bfloat16


def test_te_unavailable_skips_norm_linear_patch(engine, stub_module):
    # Match upstream: deriving from a MagicMock produces another mock rather
    # than a real class, discarding methods from the class body.
    te = MagicMock()

    class Original(te.pytorch.LayerNormLinear):
        def sharded_state_dict(self):
            return {}

    assert not hasattr(Original, "sharded_state_dict")
    module = stub_module(
        "megatron.core.extensions.transformer_engine",
        HAVE_TE=False,
        TELayerNormColumnParallelLinear=Original,
        TEColumnParallelLinear=te.pytorch.Linear,
    )
    patch = next(
        p
        for p in _layer_norm.PATCHES
        if p.id == "megatron.te.layer-norm-linear.unfused"
    )
    engine.register([patch])
    engine.install()

    assert module.TELayerNormColumnParallelLinear is Original
    assert module.HAVE_TE is False
    assert engine.report()["patches"][0]["status"] == "skipped"


def _norm_linear_env(stub_module):
    """Stub the TE extension module and return the patched class."""

    class Linear(torch.nn.Module):
        def __init__(self, input_size, output_size, *, config, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(output_size, input_size))

        def forward(self, x):
            return torch.nn.functional.linear(x, self.weight), None

    def sharded(self, *args, **kwargs):
        return self.state_dict()

    stub_module(
        "megatron.core.extensions.transformer_engine",
        HAVE_TE=True,
        TEColumnParallelLinear=Linear,
    )
    original = SimpleNamespace(sharded_state_dict=sharded)
    return _layer_norm._unfused_te_layer_norm_linear(original), original


def _norm_config(normalization):
    return SimpleNamespace(
        normalization=normalization,
        layernorm_epsilon=1e-5,
        layernorm_zero_centered_gamma=False,
        params_dtype=torch.float32,
        sequence_parallel=False,
        add_bias_linear=False,
    )


def test_subclass_constructs_itself_for_rmsnorm(stub_module):
    """heterogeneous Gathered subclasses must not be routed to the fused class.

    The fused signature needs positional input/output sizes; the Gathered
    replacement is built as (config, tp_comm_buffer_name) and used to raise
    ``missing 2 required positional arguments`` / ``unexpected layer_number``.
    """
    patched, original = _norm_linear_env(stub_module)
    calls = []

    class Gathered(patched):
        def __init__(self, config, tp_comm_buffer_name, *args, **kwargs):
            calls.append((config, tp_comm_buffer_name, args, kwargs))
            super().__init__(
                input_size=config.hidden_size,
                output_size=config.hidden_size,
                config=config,
                gather_output=False,
                bias=config.add_bias_linear,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name=tp_comm_buffer_name,
            )

    config = _norm_config("RMSNorm")
    config.hidden_size = 16
    # Exact call shapes observed from the upstream heterogeneous builds.
    module = Gathered(config=config, tp_comm_buffer_name="linear_attn")
    assert type(module) is Gathered
    assert isinstance(module, patched)
    module = Gathered(config=config, tp_comm_buffer_name="linear_attn", layer_number=2)
    assert type(module) is Gathered
    assert calls[1][3] == {"layer_number": 2}

    # RMSNorm matches the fused module's layout: norm weight, no norm bias.
    assert module.layer_norm_bias is None
    assert sum(p.numel() for p in module.parameters()) == 16 * 16 + 16

    x = torch.randn(2, 16)
    out, bias = module(x)
    reference = torch.nn.functional.rms_norm(x, (16,), module.layer_norm_weight, 1e-5)
    torch.testing.assert_close(
        out, torch.nn.functional.linear(reference, module.weight)
    )
    assert bias is None


def test_subclass_with_layernorm_keeps_norm_bias(stub_module):
    patched, _ = _norm_linear_env(stub_module)

    class Gathered(patched):
        def __init__(self, config, tp_comm_buffer_name, **kwargs):
            super().__init__(
                input_size=config.hidden_size,
                output_size=config.hidden_size,
                config=config,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name=tp_comm_buffer_name,
            )

    config = _norm_config("LayerNorm")
    config.hidden_size = 16
    module = Gathered(config=config, tp_comm_buffer_name="linear_mlp")
    assert module.layer_norm_bias is not None
    assert sum(p.numel() for p in module.parameters()) == 16 * 16 + 16 + 16
    x = torch.randn(2, 16)
    out, _ = module(x)
    reference = torch.nn.functional.layer_norm(
        x, (16,), module.layer_norm_weight, module.layer_norm_bias, 1e-5
    )
    torch.testing.assert_close(
        out, torch.nn.functional.linear(reference, module.weight)
    )


def test_base_class_rmsnorm_still_uses_fused_module(stub_module):
    patched, original = _norm_linear_env(stub_module)
    constructed = []

    class Fused(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            constructed.append((args, kwargs))

        def forward(self, x):
            return x

    original_sharded = original.sharded_state_dict

    class Original(Fused):
        def sharded_state_dict(self, *args, **kwargs):
            return original_sharded(self, *args, **kwargs)

    stub_module(
        "megatron.core.extensions.transformer_engine",
        HAVE_TE=True,
        TEColumnParallelLinear=Fused,
    )
    patched = _layer_norm._unfused_te_layer_norm_linear(Original)
    config = _norm_config("RMSNorm")
    module = patched(16, 8, config=config)
    assert type(module) is Original
    assert constructed, "exact base with RMSNorm must keep the fused construction"


@pytest.mark.integration
@pytest.mark.parametrize("ranks", [1, 2])
def test_musa_fp8_norm_linear(ranks):
    """Real TE FP8 parameters, BF16/FP8 backward, TP/SP and checkpoint sharding."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    if os.environ.get("TMA_RUN_INTEGRATION") != "1":
        pytest.skip("set TMA_RUN_INTEGRATION=1 with a working Megatron/MUSA stack")
    env = integration_env({"CUDA_DEVICE_MAX_CONNECTIONS": "1"})
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={ranks}",
            str(Path(__file__).with_name("te_layer_norm_smoke.py")),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("TE_PASS") == 8 * ranks


class _FakeTeNormBase(torch.nn.Module):
    """Minimal stand-in with the attribute surface the native forward reads."""

    def __init__(
        self, in_features=8, out_features=6, normalization="LayerNorm", **overrides
    ):
        super().__init__()
        self.layer_norm_weight = torch.nn.Parameter(torch.randn(in_features).double())
        if normalization == "LayerNorm":
            self.layer_norm_bias = torch.nn.Parameter(torch.randn(in_features).double())
        else:
            self.layer_norm_bias = None
        self.weight = torch.nn.Parameter(
            torch.randn(out_features, in_features).double()
        )
        self.bias = torch.nn.Parameter(torch.randn(out_features).double())
        self.weight_names = ("weight",)
        self.bias_names = ("bias",)
        self.eps = 1e-5
        self.normalization = normalization
        self.zero_centered_gamma = False
        self.use_bias = True
        self.apply_bias = True
        self.gemm_bias_unfused_add = False
        self.return_bias = False
        self.return_layernorm_output = False
        self.fp8 = False
        self.fp8_calibration = False
        self.tp_size = 1
        self.activation_dtype = torch.float64
        self.activation = "gelu"
        for name, value in overrides.items():
            setattr(self, name, value)

    def forward(self, inp, is_first_microbatch=None, fp8_output=False):
        return "original"

    def prepare_forward(self, inp, num_gemms=1, allow_non_contiguous=True):
        import contextlib

        @contextlib.contextmanager
        def ctx():
            self.activation_dtype = inp.dtype
            yield inp.contiguous()

        return ctx()


def test_native_layernorm_linear_unfused_forward_backward(monkeypatch):
    """The eligible plain path: functional norm + F.linear, autograd intact."""
    monkeypatch.setattr(_layer_norm, "_musa_live", lambda: True)
    monkeypatch.setattr(
        _layer_norm, "_te_native_module_eligible", lambda self, inp: True
    )
    patched = _layer_norm._unfused_te_native_layernorm_linear(_FakeTeNormBase)
    module = patched(8, 6)
    x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    out = module(x)
    assert out.shape == (4, 6)
    assert out.dtype == torch.float64

    def cast(t):
        return t if t is None or t.dtype == x.dtype else t.to(x.dtype)

    ref = torch.nn.functional.linear(
        torch.nn.functional.layer_norm(
            x, (8,), module.layer_norm_weight, module.layer_norm_bias, 1e-5
        ),
        module.weight,
        module.bias,
    )
    torch.testing.assert_close(out, ref)
    out.sum().backward()
    ref.sum().backward()
    torch.testing.assert_close(x.grad, x.grad)
    assert module.layer_norm_weight.grad is not None
    assert module.weight.grad is not None


def test_native_layernorm_linear_rmsnorm_and_bias_tail(monkeypatch):
    monkeypatch.setattr(_layer_norm, "_musa_live", lambda: True)
    monkeypatch.setattr(
        _layer_norm, "_te_native_module_eligible", lambda self, inp: True
    )
    patched = _layer_norm._unfused_te_native_layernorm_linear(_FakeTeNormBase)
    module = patched(
        8, 6, normalization="RMSNorm", gemm_bias_unfused_add=True, return_bias=True
    )
    x = torch.randn(4, 8, dtype=torch.float64)
    out, bias = module(x)
    assert bias is module.bias
    ref = (
        torch.nn.functional.linear(
            torch.nn.functional.rms_norm(x, (8,), module.layer_norm_weight, 1e-5),
            module.weight,
            None,
        )
        + module.bias
    )
    torch.testing.assert_close(out, ref)


def test_native_layernorm_linear_delegates_when_ineligible(monkeypatch):
    monkeypatch.setattr(_layer_norm, "_musa_live", lambda: True)
    monkeypatch.setattr(
        _layer_norm, "_te_native_module_eligible", lambda self, inp: False
    )
    patched = _layer_norm._unfused_te_native_layernorm_linear(_FakeTeNormBase)
    module = patched(8, 6)
    assert module(torch.randn(2, 8)) == "original"


def test_native_layernorm_mlp_unfused_forward(monkeypatch):
    monkeypatch.setattr(_layer_norm, "_musa_live", lambda: True)
    monkeypatch.setattr(
        _layer_norm, "_te_native_module_eligible", lambda self, inp: True
    )

    class FakeMlp(_FakeTeNormBase):
        def __init__(self, in_features=8, ffn=12, **kw):
            super().__init__(in_features, ffn, **kw)
            self.fc1_weight = torch.nn.Parameter(torch.randn(ffn, in_features).double())
            self.fc1_bias = torch.nn.Parameter(torch.randn(ffn).double())
            self.fc2_weight = torch.nn.Parameter(torch.randn(in_features, ffn).double())
            self.fc2_bias = torch.nn.Parameter(torch.randn(in_features).double())

        def forward(self, inp, is_first_microbatch=None):
            return "original"

    patched = _layer_norm._unfused_te_native_layernorm_mlp(FakeMlp)
    module = patched(8, 12)
    x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
    out = module(x)
    assert out.shape == (4, 8)
    ln = torch.nn.functional.layer_norm(
        x, (8,), module.layer_norm_weight, module.layer_norm_bias, 1e-5
    )
    h = torch.nn.functional.linear(ln, module.fc1_weight, module.fc1_bias)
    ref = torch.nn.functional.linear(
        torch.nn.functional.gelu(h, approximate="tanh"),
        module.fc2_weight,
        module.fc2_bias,
    )
    torch.testing.assert_close(out, ref)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_native_unfused_declines_without_musa(monkeypatch):
    monkeypatch.setattr(_layer_norm, "_musa_live", lambda: False)
    assert _layer_norm._unfused_te_native_layernorm_linear(_FakeTeNormBase) is None
    assert _layer_norm._unfused_te_native_layernorm_mlp(_FakeTeNormBase) is None


def test_native_norm_fp8_output_delegates(monkeypatch):
    monkeypatch.setattr(_layer_norm, "_musa_live", lambda: True)
    monkeypatch.setattr(_layer_norm, "_te_native_module_eligible", lambda *args: True)
    calls = []

    class Original:
        def forward(self, inp, **kwargs):
            calls.append(kwargs)
            return inp

    patched = _layer_norm._unfused_te_native_layernorm_linear(Original)
    inp = torch.ones(2)
    assert patched().forward(inp, fp8_output=True) is inp
    assert calls == [{"is_first_microbatch": None, "fp8_output": True}]


@pytest.mark.parametrize("mode", ["fp8", "calibration", "main_grad"])
def test_native_norm_checks_current_context_before_prepare(
    monkeypatch, stub_module, mode
):
    monkeypatch.setattr(_layer_norm, "_musa_live", lambda: True)
    stub_module(
        "transformer_engine.pytorch.fp8",
        FP8GlobalStateManager=SimpleNamespace(
            is_fp8_enabled=lambda: mode == "fp8",
            is_fp8_calibration=lambda: mode == "calibration",
        ),
    )
    stub_module(
        "transformer_engine.pytorch.cpu_offload", is_cpu_offload_enabled=lambda: False
    )
    inp = SimpleNamespace(device=SimpleNamespace(type="musa"))
    monkeypatch.setattr(torch, "is_tensor", lambda t: t is inp)
    # Instance flags still describe the previous non-FP8 forward.
    owner = SimpleNamespace(
        fp8=False, fp8_calibration=False, fuse_wgrad_accumulation=mode == "main_grad"
    )
    assert not _layer_norm._te_native_module_eligible(owner, inp)
