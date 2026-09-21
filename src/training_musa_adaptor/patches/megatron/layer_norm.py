"""PyTorch normalization fallback for the affected apex/TE integration.

Migrated from megatron-musa-patch ``patches/_layer_norm.py`` (rev a1090de).
The retired per-patch env switches map to ONLY/DISABLE on the patch IDs:
``TE_FUSED_LAYERNORM=1``/``TE_NORM=1``/``BLOCK_LAYERNORM=upstream`` meant
"decline this patch", which v2.0 expresses as
``TRAINING_MUSA_ADAPTOR_DISABLE=<patch id>``.

An importable apex package does not establish that its fused_layer_norm_cuda
extension can run on MUSA. The block-level TE norm also hit an allocateSpace
assertion in the bring-up stack. These are version-specific failures, not a
claim that every MUSA apex/TE build is unsupported. TE's fused norm-linear is
patched separately: replacing standalone norms cannot reach its internal norm.
Functional PyTorch ops keep
the module/parameter contract, but kernel choice, rounding and performance can
differ from fused implementations and must be checked on stack upgrades.
"""

from __future__ import annotations

import numbers
from typing import Any

from ..._engine import AttrPatch
from ...backends import musa_available as _musa_live

__all__ = ["PATCHES"]


def _te_norm_compute_dtype(torch, params_dtype):
    """TE's effective norm dtype: ``maybe_autocast_dtype(default=weight.dtype)``.

    TransformerEngine's basic norm ``op_forward`` resolves the compute dtype
    from the *parameters* (or the autocast dtype when one is active) and casts
    the *input* to it -- ``x = reshape(input_, ..., dtype=dtype)`` in
    ``pytorch/ops/basic/layer_norm.py``. That contract is what absorbs a
    mixed-dtype activation (e.g. a bf16 model fed the fp32 learned
    position-embedding sum: ``torch.nn.Embedding`` defaults to fp32 while
    ``VocabParallelEmbedding`` honors ``params_dtype``) before the next TE
    linear asserts ``input.dtype == param.dtype``. Casting the other way --
    parameters to the input's dtype -- leaks the wider dtype downstream
    instead (observed as "Data types for parameters must match when outside of
    autocasted region" in the a2a overlap suite).
    """
    if torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return params_dtype


def _build_norm_fallback_class(base: Any = None, cast_to_input: bool = False) -> Any:
    """The functional LayerNorm/RMSNorm class shared by the norm patches.

    ``base`` (when given) is Megatron's local FusedLayerNorm; subclassing it
    keeps ``isinstance`` checks and any consumer-authored class attributes.
    ``cast_to_input`` mirrors TE's norm contract: parameters stay in their
    created dtype while the output takes the input's dtype, so consumers that
    feed the norm result straight into a TE op never see a dtype mismatch.
    """
    import torch

    bases = (base,) if base is not None else (torch.nn.Module,)

    class FusedLayerNorm(*bases):
        """Match Megatron's norm constructor and optimizer parameter markers.

        Config is authoritative, as in upstream FusedLayerNorm/TENorm. In
        particular TransformerBlock passes config.normalization but omits the
        normalization keyword, so using that keyword's default corrupts RMSNorm
        models. Legacy constructor keywords remain accepted for compatibility.
        """

        _megatron_musa_patch_fallback = True
        _cast_output_to_input = cast_to_input

        def __init__(
            self,
            config,
            hidden_size,
            eps: float = 1e-5,
            persist_layer_norm: bool = True,
            zero_centered_gamma: bool = False,
            normalization: str = "LayerNorm",
        ):
            super().__init__()
            self.config = config
            self.zero_centered_gamma = config.layernorm_zero_centered_gamma
            self.normalization = getattr(config, "normalization", normalization)
            if self.normalization not in {"LayerNorm", "RMSNorm"}:
                raise ValueError(f"Unsupported normalization: {self.normalization!r}")
            if isinstance(hidden_size, numbers.Integral):
                hidden_size = (hidden_size,)
            self.hidden_size = torch.Size(hidden_size)
            self.eps = eps
            # This fallback never selects apex's persistent FastLayerNorm kernel.
            self.persist_layer_norm = False

            params_dtype = getattr(config, "params_dtype", None)
            empty = (
                (lambda *shape: torch.empty(*shape, dtype=params_dtype))
                if params_dtype is not None
                else torch.empty
            )

            self.weight = torch.nn.Parameter(empty(self.hidden_size))
            if self.normalization == "LayerNorm":
                self.bias = torch.nn.Parameter(empty(self.hidden_size))
            else:
                self.register_parameter("bias", None)
            self.reset_parameters()

            self.sequence_parallel = config.sequence_parallel
            self.weight.sequence_parallel = self.sequence_parallel
            if self.bias is not None:
                self.bias.sequence_parallel = self.sequence_parallel

        def reset_parameters(self) -> None:
            torch.nn.init.constant_(self.weight, 0 if self.zero_centered_gamma else 1)
            if self.bias is not None:
                torch.nn.init.zeros_(self.bias)

        def forward(self, input):
            weight = self.weight + 1 if self.zero_centered_gamma else self.weight
            bias = self.bias
            if self._cast_output_to_input and weight.dtype != input.dtype:
                weight = weight.to(input.dtype)  # type: ignore[assignment]
                bias = bias.to(input.dtype) if bias is not None else None  # type: ignore[assignment]
            if self.normalization == "RMSNorm":
                return torch.nn.functional.rms_norm(input, self.hidden_size, weight, self.eps)
            return torch.nn.functional.layer_norm(input, self.hidden_size, weight, bias, self.eps)

    return FusedLayerNorm


def _pure_torch_layer_norm(original: Any) -> Any:
    """Build a fallback without importing torch until the target is available."""
    return _build_norm_fallback_class()


def _using_torch_fallback() -> bool:
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    return bool(getattr(FusedLayerNorm, "_megatron_musa_patch_fallback", False))


def _fallback_available_flag(original: Any) -> Any:
    # ONLY/DISABLE may select flags independently; never advertise a fallback
    # if its class replacement was not applied.
    return True if _using_torch_fallback() else None


def _persistent_available_flag(original: Any) -> Any:
    return False if _using_torch_fallback() else None


def _block_layer_norm_impl(original: Any) -> Any:
    """Use a functional norm even when the separate local-class patch is off."""
    from megatron.core.fusions.fused_layer_norm import FusedLayerNorm

    if getattr(FusedLayerNorm, "_megatron_musa_patch_fallback", False):
        return FusedLayerNorm
    return _pure_torch_layer_norm(FusedLayerNorm)


def _unfused_te_layer_norm_linear(original: Any) -> Any:
    """Unfuse LayerNorm only; construct native TE modules for RMSNorm.

    Register norm parameters on the linear itself to retain the fused module's
    checkpoint names and replicated (rather than column-sharded) norm weights.
    """
    from megatron.core.extensions.transformer_engine import HAVE_TE

    # Megatron uses MagicMock-backed placeholders when TE imports fail. They
    # are not usable module classes and have no sharded_state_dict to reuse.
    # Keep upstream's optional-dependency behavior in that case.
    if not HAVE_TE:
        return None
    import torch
    from megatron.core.extensions.transformer_engine import TEColumnParallelLinear

    # te.pytorch.LayerNormLinear must still accept this class in runtime
    # isinstance checks (TEFusedMLP rejects an FC1 that is not a TE
    # LayerNormLinear). Inheriting it as a second base is not an option: the
    # MRO would route te.pytorch.Linear's super() into LayerNormLinear's
    # constructor. Megatron's own class derives from it; this replacement
    # registers as a virtual subclass of the *current* binding instead (TE
    # modules are ABCs), which stays true whether that binding is upstream or
    # this package's native-unfused subclass. Resolved defensively: CPU stub
    # tests build this class without a usable real TE import.
    te_layernorm_linear = None
    try:
        import transformer_engine as _te

        te_layernorm_linear = getattr(getattr(_te, "pytorch", None), "LayerNormLinear", None)
    except Exception:  # noqa: BLE001 - stubbed or broken TE skips registration
        te_layernorm_linear = None

    class _NormLinearDispatchMeta(type(TEColumnParallelLinear)):
        def __instancecheck__(cls, instance):
            # Megatron's FP8 parameter gathering identifies column-parallel
            # modules with isinstance(..., TELayerNormColumnParallelLinear).
            return super().__instancecheck__(instance) or isinstance(instance, original)

    class TELayerNormColumnParallelLinear(
        TEColumnParallelLinear, metaclass=_NormLinearDispatchMeta
    ):
        _megatron_musa_patch_fallback = True

        def __new__(cls, *args, **kwargs):
            # Dispatch once at construction. Returning the original instance
            # keeps RMSNorm's fused forward/backward and adds no runtime wrapper.
            # Other normalization values retain upstream validation as well.
            # deepcopy reconstructs modules via __new__(cls) without config.
            #
            # Only the exact base class may hand construction to the fused
            # module. Subclasses (e.g. the heterogeneous
            # TELayerNormColumnParallelLinearGathered) are built with a
            # (config, tp_comm_buffer_name) signature the fused __init__ cannot
            # accept; they must run their own constructor, which forwards real
            # input/output sizes to this class.
            config = kwargs.get("config")
            if (
                cls is _BASE_NORM_LINEAR
                and config is not None
                and config.normalization != "LayerNorm"
            ):
                return original(*args, **kwargs)
            return super().__new__(cls)

        def __init__(self, input_size, output_size, *, config, **kwargs):
            if kwargs.get("is_expert", False):
                raise ValueError("Transformer Engine norm-linear layers do not support MoE")
            if config.normalization not in ("LayerNorm", "RMSNorm"):
                raise ValueError(f"Unsupported normalization: {config.normalization!r}")
            super().__init__(input_size, output_size, config=config, **kwargs)
            self.normalization = config.normalization
            self.eps = config.layernorm_epsilon
            self.zero_centered_gamma = config.layernorm_zero_centered_gamma
            self.layer_norm_weight = torch.nn.Parameter(
                torch.full(
                    (input_size,),
                    0.0 if self.zero_centered_gamma else 1.0,
                    dtype=config.params_dtype,
                    device=self.weight.device,
                )
            )
            if self.normalization == "LayerNorm":
                self.layer_norm_bias = torch.nn.Parameter(
                    torch.zeros(
                        input_size,
                        dtype=config.params_dtype,
                        device=self.weight.device,
                    )
                )
            else:
                # Match the fused module's RMSNorm layout: no norm bias.
                self.register_parameter("layer_norm_bias", None)
            for parameter in (self.layer_norm_weight, self.layer_norm_bias):
                if parameter is not None:
                    parameter.sequence_parallel = config.sequence_parallel
                    parameter.allreduce = True

        def forward(self, x):
            # TE's fused op chain normalizes in
            # maybe_autocast_dtype(default=norm-weight dtype) and casts the
            # input to it; the following linear therefore sees the parameter
            # dtype outside autocast. Mirror that here so a mixed-dtype
            # activation (fp32 position-embedding sum into a bf16 model) is
            # absorbed at the norm exactly like the native kernel does.
            dtype = _te_norm_compute_dtype(torch, self.layer_norm_weight.dtype)
            if x.dtype != dtype:
                x = x.to(dtype)
            weight = self.layer_norm_weight
            if dtype != weight.dtype:
                weight = weight.to(dtype)
            bias = self.layer_norm_bias
            if bias is not None and dtype != bias.dtype:
                bias = bias.to(dtype)
            weight = weight + 1 if self.zero_centered_gamma else weight
            if self.normalization == "RMSNorm":
                normalized = torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, self.eps)
            else:
                normalized = torch.nn.functional.layer_norm(
                    x,
                    (x.shape[-1],),
                    weight,
                    bias,
                    self.eps,
                )
            return super().forward(normalized)

        # The upstream fused wrapper already handles metadata defaults and
        # shards only weight/bias, leaving layer_norm_* replicated.
        sharded_state_dict = original.sharded_state_dict

    # Resolved after the class statement so __new__ can tell the exact base
    # from subclasses without relying on the module attribute (which upstream
    # consumers may rebind).
    _BASE_NORM_LINEAR = TELayerNormColumnParallelLinear

    if isinstance(te_layernorm_linear, type) and hasattr(te_layernorm_linear, "register"):
        te_layernorm_linear.register(TELayerNormColumnParallelLinear)

    return TELayerNormColumnParallelLinear


def _te_native_module_eligible(self, inp) -> bool:
    """Whether TE's fused norm module can be replaced by the functional path.

    Only the plain path is claimed: a live MUSA device, no FP8 (weights or
    calibration), no user-buffer overlap, no CPU offloading, no TP
    communication (tp_size 1), no layernorm-output return and a supported
    normalization. Everything else keeps the original TE forward so its own
    errors surface.
    """
    import sys

    import torch

    if not _musa_live() or not torch.is_tensor(inp) or inp.device.type != "musa":
        return False
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    # prepare_forward refreshes these flags; instance state describes the
    # previous call, so inspect the current context before choosing a path.
    if (
        FP8GlobalStateManager.is_fp8_enabled()
        or FP8GlobalStateManager.is_fp8_calibration()
        or getattr(self, "fuse_wgrad_accumulation", False)
    ):
        return False
    if getattr(self, "fp8", False) or getattr(self, "fp8_calibration", False):
        return False
    if getattr(self, "tp_size", 1) > 1:
        return False
    if any(
        getattr(self, name, False)
        for name in (
            "ub_overlap_ag_fprop",
            "ub_overlap_rs_fprop",
            "ub_overlap_ag_dgrad",
            "ub_overlap_rs_dgrad",
            "ub_bulk_wgrad",
            "ub_bulk_dgrad",
            "ub_overlap_ag",
            "ub_overlap_rs",
            "ub_overlap_rs_dgrad",
        )
    ):
        return False
    if getattr(self, "return_layernorm_output", False):
        return False
    offload = sys.modules.get("transformer_engine.pytorch.cpu_offload")
    if offload is not None and offload.is_cpu_offload_enabled():
        return False
    if getattr(self, "normalization", "LayerNorm") not in ("LayerNorm", "RMSNorm"):
        return False
    return True


def _te_plain_norm(self, x, activation_dtype):
    """The module's normalization as a functional op, TE's dtype contract."""
    import torch.nn.functional as F

    def cast(t):
        return t if t is None or t.dtype == activation_dtype else t.to(activation_dtype)

    weight = cast(self.layer_norm_weight)
    if getattr(self, "zero_centered_gamma", False):
        weight = weight + 1
    bias = cast(self.layer_norm_bias)
    in_features = self.layer_norm_weight.numel()
    if self.normalization == "LayerNorm":
        return F.layer_norm(x, (in_features,), weight, bias, self.eps)
    return F.rms_norm(x, (in_features,), weight, self.eps)


def _te_plain_weight(self, name_base, activation_dtype):
    """Concatenated (and cast) split parameters, mirroring the module's noop_cat."""
    import sys

    import torch

    quantized = getattr(sys.modules.get("transformer_engine.pytorch.tensor"), "QuantizedTensor", ())
    tensors = [getattr(self, name) for name in getattr(self, f"{name_base}_names")]
    tensors = [t.dequantize() if isinstance(t, quantized) else t for t in tensors]
    weight = tensors[0] if len(tensors) == 1 else torch.cat(tensors)
    if weight.dtype != activation_dtype:
        weight = weight.to(activation_dtype)
    return weight


def _te_plain_bias(self, activation_dtype):
    """Concatenated bias parameters, or the unused placeholder like upstream."""

    def cast(t):
        return t if t is None or t.dtype == activation_dtype else t.to(activation_dtype)

    if self.use_bias:
        import torch

        biases = [getattr(self, name) for name in self.bias_names]
        return cast(biases[0] if len(biases) == 1 else torch.cat(biases))
    return cast(getattr(self, self.bias_names[0]))


def _unfused_te_native_layernorm_linear(original: Any) -> Any:
    """Subclass te.pytorch.LayerNormLinear with a functional plain forward.

    Direct TE models (``te.pytorch.TransformerLayer`` in the megatron-FSDP
    suite) run the fused normalization through
    ``module/_common.py:apply_normalization``, whose MUSA kernel aborts in
    ``csrc/common.cpp allocateSpace``. The replacement subclasses the TE
    module -- isinstance checks, parameters, checkpoint metadata and the
    constructor stay TE's -- and only reroutes the *eligible* plain path
    (see ``_te_native_module_eligible``) through functional normalization
    plus ``F.linear``, so both forward and backward run on plain PyTorch ops.
    """
    if not _musa_live():
        return None

    class LayerNormLinearUnfused(original):  # type: ignore[misc,valid-type]
        _megatron_musa_patch_fallback = True

        def forward(self, inp, is_first_microbatch=None, fp8_output=False):
            if fp8_output or not _te_native_module_eligible(self, inp):
                return original.forward(
                    self, inp, is_first_microbatch=is_first_microbatch, fp8_output=fp8_output
                )
            import torch.nn.functional as F

            with self.prepare_forward(inp, allow_non_contiguous=False) as x:
                if x.dtype != self.activation_dtype:
                    x = x.to(self.activation_dtype)
                ln_out = _te_plain_norm(self, x, self.activation_dtype)
                weight = _te_plain_weight(self, "weight", self.activation_dtype)
                bias_tensor = _te_plain_bias(self, self.activation_dtype)
                gemm_bias = (
                    bias_tensor if self.apply_bias and not self.gemm_bias_unfused_add else None
                )
                out = F.linear(ln_out, weight, gemm_bias)
            if self.gemm_bias_unfused_add:
                out = out + bias_tensor
            if self.return_bias:
                return out, bias_tensor
            return out

    return LayerNormLinearUnfused


def _unfused_te_native_layernorm_mlp(original: Any) -> Any:
    """Subclass te.pytorch.LayerNormMLP with a functional plain forward.

    Same fused-normalization failure and the same contract as the
    LayerNormLinear replacement; the eligible path is norm -> fc1 -> gelu/relu
    -> fc2 on plain PyTorch ops. Gated activations keep the original forward.
    """
    if not _musa_live():
        return None

    class LayerNormMLPUnfused(original):  # type: ignore[misc,valid-type]
        _megatron_musa_patch_fallback = True

        def forward(self, inp, is_first_microbatch=None):
            if not _te_native_module_eligible(self, inp):
                return original.forward(self, inp, is_first_microbatch=is_first_microbatch)
            if self.activation not in ("gelu", "relu"):
                return original.forward(self, inp, is_first_microbatch=is_first_microbatch)
            import torch.nn.functional as F

            with self.prepare_forward(inp, num_gemms=2) as x:
                if x.dtype != self.activation_dtype:
                    x = x.to(self.activation_dtype)
                ln_out = _te_plain_norm(self, x, self.activation_dtype)
                fc1_weight = self.fc1_weight
                fc2_weight = self.fc2_weight
                if fc1_weight.dtype != self.activation_dtype:
                    fc1_weight = fc1_weight.to(self.activation_dtype)
                if fc2_weight.dtype != self.activation_dtype:
                    fc2_weight = fc2_weight.to(self.activation_dtype)
                fc1_bias = self.fc1_bias if self.use_bias else None
                if fc1_bias is not None and fc1_bias.dtype != self.activation_dtype:
                    fc1_bias = fc1_bias.to(self.activation_dtype)
                fc2_bias = self.fc2_bias if self.use_bias else None
                if fc2_bias is not None and fc2_bias.dtype != self.activation_dtype:
                    fc2_bias = fc2_bias.to(self.activation_dtype)
                fc1_out = F.linear(ln_out, fc1_weight, fc1_bias)
                # TE fuses gelu into the GEMM; the tanh approximation matches
                # tex.gelu's formula.
                if self.activation == "gelu":
                    act_out = F.gelu(fc1_out, approximate="tanh")
                else:
                    act_out = F.relu(fc1_out)
                gemm2_bias = (
                    fc2_bias if self.apply_bias and not self.gemm_bias_unfused_add else None
                )
                out = F.linear(act_out, fc2_weight, gemm2_bias)
            if self.gemm_bias_unfused_add:
                out = out + fc2_bias
            if self.return_bias:
                return out, fc2_bias
            return out

    return LayerNormMLPUnfused


def _unfused_te_fused_mlp(original: Any) -> Any:
    """Subclass Megatron's TEFusedMLP with the plain MLP decomposition.

    TEFusedMLP runs its norm+fc1+activation+fc2 chain through TE's
    operation-fuser (``BasicLayerNorm``/``BasicLinear`` ops), whose
    normalization aborts in the same MUSA allocateSpace assertion as the
    fused modules. The plain ``MLP.forward`` decomposition -- already used
    everywhere else -- runs linear_fc1 (the unfused norm-linear replacement),
    the functional activation and linear_fc2 on regular PyTorch ops.
    """

    if not _musa_live():
        return None
    from megatron.core.transformer.mlp import MLP

    class TEFusedMLPUnfused(original):  # type: ignore[misc,valid-type]
        _megatron_musa_patch_fallback = True

        def forward(self, hidden_states, **kwargs):
            config = getattr(self, "config", None)
            if config is not None and getattr(config, "fp8", False):
                # FP8 needs TE's fused quantization; keep upstream so its own
                # errors surface instead of silently changing precision.
                return original.forward(self, hidden_states, **kwargs)
            import torch

            if not torch.is_tensor(hidden_states) or hidden_states.device.type != "musa":
                return original.forward(self, hidden_states, **kwargs)
            return MLP.forward(self, hidden_states, **kwargs)

    return TEFusedMLPUnfused


def _te_norm_unfused(original: Any) -> Any:
    """Build TENorm's norm with functional PyTorch ops on MUSA.

    ``TENorm.__new__`` constructs ``te.pytorch.LayerNorm``/``RMSNorm``, whose
    standalone norm op aborts on the affected MUSA stack (allocateSpace
    assertion in transformer_engine ops/basic/layer_norm.py). The replacement
    subclasses those TE modules -- so ``isinstance`` checks, parameter names,
    dtypes, initialization and ``sharded_state_dict`` stay TE's -- and only
    replaces ``forward`` with the functional implementation. The dtype
    contract matches TE's ``op_forward``: the compute dtype is the autocast
    dtype when autocast is active and the parameter dtype otherwise, and the
    *input* is cast to it (never the reverse), so mixed-dtype activations are
    absorbed exactly like the native kernel. TE_NORM=1 keeps upstream for
    upgrade validation.
    """
    import sys

    from megatron.core.extensions.transformer_engine import HAVE_TE

    if not HAVE_TE:
        # Keep upstream's optional-dependency error.
        return None
    if not _musa_live():
        return None

    import numbers

    import torch
    import transformer_engine as te

    module = sys.modules["megatron.core.extensions.transformer_engine"]
    extra_kwargs_fn = getattr(module, "_get_extra_te_kwargs", None)

    def build(te_cls, normalization):
        class TENormFallback(te_cls):  # type: ignore[valid-type,misc]
            _megatron_musa_patch_fallback = True

            def __init__(self, config, hidden_size, eps=1e-5):
                te_cls.__init__(
                    self,
                    hidden_size=hidden_size,
                    eps=eps,
                    sequence_parallel=config.sequence_parallel,
                    zero_centered_gamma=config.layernorm_zero_centered_gamma,
                    **(extra_kwargs_fn(config) if extra_kwargs_fn else {}),
                )
                self.config = config
                self.normalization = normalization
                self.hidden_size = torch.Size(
                    (hidden_size,) if isinstance(hidden_size, numbers.Integral) else hidden_size  # type: ignore[arg-type]
                )

            def forward(self, input):
                # Same contract as TE's basic norm op_forward: compute in
                # maybe_autocast_dtype(default=weight.dtype), casting the
                # input (never promoting the parameters to the input's
                # dtype). The output therefore leaves in the parameter dtype,
                # absorbing mixed-dtype activations before the next TE module.
                dtype = _te_norm_compute_dtype(torch, self.weight.dtype)
                if input.dtype != dtype:
                    input = input.to(dtype)
                weight = self.weight
                if dtype != weight.dtype:
                    weight = weight.to(dtype)
                if getattr(self, "zero_centered_gamma", False):
                    weight = weight + 1
                if normalization == "LayerNorm":
                    bias = self.bias
                    if bias is not None and dtype != bias.dtype:
                        bias = bias.to(dtype)
                    return torch.nn.functional.layer_norm(
                        input, self.hidden_size, weight, bias, self.eps
                    )
                return torch.nn.functional.rms_norm(input, self.hidden_size, weight, self.eps)

        TENormFallback.__name__ = f"TENorm{normalization}Fallback"
        return TENormFallback

    layernorm_fallback = build(te.pytorch.LayerNorm, "LayerNorm")
    rmsnorm_fallback = build(te.pytorch.RMSNorm, "RMSNorm")

    class TENormMusa:
        """Drop-in for upstream's TENorm conditional wrapper."""

        _megatron_musa_patch_fallback = True

        def __new__(cls, config, hidden_size, eps: float = 1e-5):
            normalization = getattr(config, "normalization", "LayerNorm")
            if normalization == "LayerNorm":
                return layernorm_fallback(config, hidden_size, eps)
            if normalization == "RMSNorm":
                assert hasattr(
                    te.pytorch, "RMSNorm"
                ), "Transformer-Engine >= v0.11 required to use this feature"
                return rmsnorm_fallback(config, hidden_size, eps)
            raise Exception("Only LayerNorm and RMSNorm are curently supported")

    return TENormMusa


PATCHES = (
    AttrPatch(
        id="transformer_engine.layer-norm-linear.native-unfused",
        version_gates=("transformer_engine >=2.0,<2.1",),
        target="transformer_engine.pytorch.module.layernorm_linear:LayerNormLinear",
        rebind_prefixes=("transformer_engine",),
        replace=_unfused_te_native_layernorm_linear,
        rationale=(
            "Direct TransformerEngine models (te.pytorch.TransformerLayer in "
            "the megatron-FSDP suite) normalize inside "
            "module/_common.py:apply_normalization, whose MUSA kernel aborts "
            "in csrc/common.cpp allocateSpace ('Should never reach here'). "
            "56 mfsdp cases fail this way; the megatron-scope "
            "TELayerNormColumnParallelLinear patch cannot reach modules TE "
            "constructs itself."
        ),
        strategy=(
            "Subclass te.pytorch.LayerNormLinear so isinstance checks, "
            "parameters, quantizer state and checkpoint metadata stay TE's, "
            "and reroute only the eligible plain path (live MUSA input, no "
            "FP8/calibration, no user-buffer overlap, no CPU offloading, "
            "tp_size 1, no layernorm-output return, LayerNorm/RMSNorm) "
            "through functional normalization plus F.linear on the split "
            "parameters -- forward and backward both run on plain PyTorch "
            "ops, bypassing the fused kernel and its custom autograd node. "
            "The norm follows TE's dtype contract (input cast to the "
            "activation/parameter dtype). Everything else calls the original "
            "forward so its own errors surface. TE_FUSED_LAYERNORM=1 "
            "restores upstream for upgrade validation."
        ),
        upstream=(
            "TransformerEngine pytorch/module/layernorm_linear.py:"
            "LayerNormLinear.forward; module/_common.py:apply_normalization"
        ),
        remove_when=(
            "Remove after the MUSA TransformerEngine fused LayerNormLinear "
            "native path passes the megatron-FSDP te_transformer cases "
            "(test_mfsdp_fully_shard and the DCP round-trips) on MUSA with "
            "this patch disabled."
        ),
    ),
    AttrPatch(
        id="transformer_engine.layer-norm-mlp.native-unfused",
        version_gates=("transformer_engine >=2.0,<2.1",),
        target="transformer_engine.pytorch.module.layernorm_mlp:LayerNormMLP",
        rebind_prefixes=("transformer_engine",),
        replace=_unfused_te_native_layernorm_mlp,
        rationale=(
            "te.pytorch.TransformerLayer's MLP block normalizes through the "
            "same module/_common.py:apply_normalization path as "
            "LayerNormLinear and aborts in the same MUSA allocateSpace "
            "assertion."
        ),
        strategy=(
            "Subclass te.pytorch.LayerNormMLP with the same eligibility and "
            "contract as the LayerNormLinear replacement; the eligible plain "
            "path is functional norm -> fc1 -> gelu(tanh, matching tex.gelu) "
            "or relu -> fc2 on plain PyTorch ops. Gated activations and "
            "every non-plain configuration keep the original forward so "
            "their own errors surface. TE_FUSED_LAYERNORM=1 restores "
            "upstream for upgrade validation."
        ),
        upstream=(
            "TransformerEngine pytorch/module/layernorm_mlp.py:"
            "LayerNormMLP.forward; module/_common.py:apply_normalization"
        ),
        remove_when=(
            "Remove together with "
            "transformer_engine.layer-norm-linear.native-unfused after the "
            "native fused path passes the megatron-FSDP te_transformer "
            "cases on MUSA."
        ),
    ),
    AttrPatch(
        id="megatron.te.layer-norm-linear.unfused",
        rebind_prefixes=("megatron",),
        version_gates=("transformer_engine >=2.0,<2.1",),
        target="megatron.core.extensions.transformer_engine:TELayerNormColumnParallelLinear",
        replace=_unfused_te_layer_norm_linear,
        rationale=(
            "TE GPT specs fuse QKV/FC1 normalization inside LayerNormLinear; "
            "neither local FusedLayerNorm nor block LayerNormImpl covers its "
            "MUSA allocateSpace assertion. The affected TE port comments out "
            "the LayerNorm workspace query/kernel but still allocates the "
            "uninitialized workspace shape."
        ),
        strategy=(
            "For LayerNorm only, use PyTorch normalization followed by TEColumnParallelLinear, "
            "retaining FP8, TP, norm parameter names and sharding. The norm step follows TE's "
            "op contract: it computes in maybe_autocast_dtype(default=norm-weight dtype) and "
            "casts the input to it, so a mixed-dtype activation (the fp32 learned "
            "position-embedding sum entering a bf16 model) is absorbed here instead of "
            "tripping the next TE linear's dtype assert. The exact base class with a "
            "non-LayerNorm config still constructs the original fused module; subclasses always "
            "run their own constructor (the fused signature cannot accept their "
            "(config, tp_comm_buffer_name) call shape) and this class then honors the config's "
            "RMSNorm, matching the fused module's no-bias parameter layout. "
            "TE_FUSED_LAYERNORM=1 restores upstream for upgrade validation."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/extensions/transformer_engine.py",
        remove_when=(
            "Remove after upgraded MT-TransformerEngine fused LayerNormLinear "
            "passes LayerNorm FP8 forward/backward, zero-centered gamma, "
            "checkpoint and multi-rank sequence-parallel tests on MUSA."
        ),
    ),
    AttrPatch(
        id="megatron.fusions.fused-layer-norm.pure-torch",
        rebind_prefixes=("megatron",),
        target="megatron.core.fusions.fused_layer_norm:FusedLayerNorm",
        replace=_pure_torch_layer_norm,
        rationale=(
            "In the affected stack apex imports but its fused_layer_norm_cuda "
            "extension is unavailable, so Megatron's import probe selects a class "
            "that fails on forward instead of a usable backend fallback."
        ),
        strategy=(
            "Replace only the norm class with functional PyTorch LayerNorm/RMSNorm "
            "selected from config, preserving gamma convention, state-dict keys and "
            "sequence-parallel markers; do not promise fused-kernel numerical identity."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/fusions/fused_layer_norm.py",
        remove_when=(
            "Review on apex/MT-TransformerEngine, torch_musa and Megatron upgrades; "
            "remove when upstream selects a usable norm and forward/backward, "
            "low-precision, zero-centered-gamma and checkpoint parity tests pass "
            "on MUSA. Benchmark the replacement before switching."
        ),
    ),
    AttrPatch(
        id="megatron.fusions.fused-layer-norm.have-apex-flag",
        rebind_prefixes=("megatron",),
        requires=("megatron.fusions.fused-layer-norm.pure-torch",),
        target="megatron.core.fusions.fused_layer_norm:HAVE_FUSED_LAYER_NORM",
        replace=_fallback_available_flag,
        rationale=(
            "Consumers such as bert_lm_head use HAVE_FUSED_LAYER_NORM to decide "
            "whether the local norm class is usable, even after class replacement."
        ),
        strategy=(
            "Advertise the installed fallback as usable; this flag does not "
            "certify that an apex fused kernel is available. Skip if selective "
            "patch filtering left the local class unreplaced."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/fusions/fused_layer_norm.py",
        remove_when=(
            "Remove together with the pure-torch norm patch; on Megatron upgrades "
            "review flag consumers, including the BERT head's norm construction."
        ),
    ),
    AttrPatch(
        id="megatron.fusions.persist-layer-norm.disable",
        rebind_prefixes=("megatron",),
        requires=("megatron.fusions.fused-layer-norm.pure-torch",),
        target="megatron.core.fusions.fused_layer_norm:HAVE_PERSIST_LAYER_NORM",
        replace=_persistent_available_flag,
        rationale=(
            "The functional fallback does not execute apex's FastLayerNormFN; "
            "leaving its persistent-kernel flag enabled misdescribes the selected path."
        ),
        strategy=(
            "Set the persistent-kernel availability flag false only while the "
            "functional fallback is installed; otherwise leave upstream unchanged."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/fusions/fused_layer_norm.py",
        remove_when=(
            "Remove with the functional fallback after a usable native norm is "
            "validated; review any new consumers of this flag on Megatron upgrades."
        ),
    ),
    AttrPatch(
        id="megatron.transformer-block.layer-norm.impl-local",
        rebind_prefixes=("megatron",),
        version_gates=("transformer_engine >=2.0,<2.1",),
        target="megatron.core.transformer.transformer_block:LayerNormImpl",
        replace=_block_layer_norm_impl,
        rationale=(
            "TransformerBlock chooses TENorm whenever TE imports, including for "
            "a local layer spec. The affected TE MUSA norm path aborted in "
            "allocateSpace; importability alone did not prove runtime usability."
        ),
        strategy=(
            "Bind the block's default norm to a functional fallback, reusing the "
            "patched local class when present but not requiring its patch. Explicit "
            "block submodule specs are unchanged; BLOCK_LAYERNORM=upstream keeps "
            "the upstream default for upgrade testing."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/transformer/transformer_block.py",
        remove_when=(
            "Review on MT-TransformerEngine/torch_musa and Megatron upgrades; remove "
            "after BLOCK_LAYERNORM=upstream passes LayerNorm and RMSNorm final-block "
            "forward/backward, checkpoint and multi-rank sequence-parallel tests."
        ),
    ),
    AttrPatch(
        id="megatron.te.norm.unfused-musa",
        rebind_prefixes=("megatron",),
        version_gates=("transformer_engine >=2.0,<2.1",),
        target="megatron.core.extensions.transformer_engine:TENorm",
        replace=_te_norm_unfused,
        rationale=(
            "TENorm constructs te.pytorch.LayerNorm/RMSNorm, whose standalone "
            "norm op aborts on the affected MUSA stack with the allocateSpace "
            "assertion (transformer_engine ops/basic/layer_norm.py reached from "
            "TE's csrc allocator). 44 upstream cases fail this way: DSA "
            "attention variants, MTP, LLaVA, CLIP ViT, the VLM controller and "
            "spec customization, all of which build their standalone norms "
            "through TENorm."
        ),
        strategy=(
            "Replace the TENorm wrapper with subclasses of TE's own "
            "LayerNorm/RMSNorm chosen from config.normalization, so isinstance "
            "checks, parameter names, dtypes, initialization and "
            "sharded_state_dict stay TE's, while forward runs the functional "
            "norm under TE's own dtype contract: the compute dtype is "
            "maybe_autocast_dtype(default=weight.dtype) and the *input* is "
            "cast to it, never the reverse, so mixed-dtype activations (the "
            "fp32 learned position-embedding sum entering a bf16 model) are "
            "absorbed exactly like the native kernel instead of leaking to the "
            "next TE linear. Upstream validation for unsupported normalization "
            "values is retained. Only declines (keeping upstream) when TE is "
            "absent, no MUSA runtime is live, or TE_NORM=1 requests the "
            "upgrade-validation path."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/core/extensions/transformer_engine.py:"
            "TENorm; TransformerEngine ops/basic/layer_norm.py"
        ),
        remove_when=(
            "Remove after the MUSA TransformerEngine standalone LayerNorm/RMSNorm "
            "passes forward/backward, zero-centered gamma, meta materialization, "
            "checkpoint key/sharding and multi-rank sequence-parallel tests; "
            "re-run the 44 affected node ids with TE_NORM=1 before deleting."
        ),
    ),
    AttrPatch(
        id="megatron.te.fused-mlp.unfused",
        rebind_prefixes=("megatron",),
        version_gates=("transformer_engine >=2.0,<2.1",),
        target="megatron.core.extensions.transformer_engine:TEFusedMLP",
        replace=_unfused_te_fused_mlp,
        rationale=(
            "Megatron's TEFusedMLP (the use_te_fused_ops GPT spec, e.g. "
            "models/test_gpt_model.py::TestGPTWithFusedOps) runs its "
            "norm+MLP chain through TE's operation fuser, whose BasicLayerNorm "
            "op aborts in the same MUSA allocateSpace assertion as the fused "
            "norm modules; the module-level LayerNormLinear/LayerNormMLP "
            "patches cannot reach an ops.Sequential chain."
        ),
        strategy=(
            "Subclass TEFusedMLP and reroute non-FP8 MUSA forwards to the "
            "plain MLP.forward decomposition: linear_fc1 (the unfused "
            "norm-linear replacement handles normalization), the functional "
            "activation, then linear_fc2. FP8 and non-MUSA inputs keep the "
            "original fused forward so quantization semantics and its own "
            "errors surface. TE_FUSED_LAYERNORM=1 restores upstream."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/core/extensions/transformer_engine.py:"
            "TEFusedMLP; megatron/core/transformer/mlp.py:MLP.forward"
        ),
        remove_when=(
            "Remove after the MUSA TransformerEngine operation fuser's "
            "BasicLayerNorm passes TestGPTWithFusedOps on MUSA with this "
            "patch disabled."
        ),
    ),
)
