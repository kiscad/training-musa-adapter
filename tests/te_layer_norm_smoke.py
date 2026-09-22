"""Hardware worker for test_te_layer_norm.py; run via torch.distributed.run."""

import os
from itertools import product

import torch
import transformer_engine.pytorch as te
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TELayerNormColumnParallelLinear
from megatron.core.fp8_utils import is_column_parallel_linear
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from transformer_engine.common.recipe import DelayedScaling, Format

torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
world_size = int(os.environ.get("WORLD_SIZE", 1))
torch.distributed.init_process_group("nccl")
parallel_state.initialize_model_parallel(tensor_model_parallel_size=world_size)
model_parallel_cuda_manual_seed(123)
spec = get_gpt_layer_with_transformer_engine_spec()
assert (
    spec.submodules.self_attention.submodules.linear_qkv
    is TELayerNormColumnParallelLinear
)
assert spec.submodules.mlp.submodules.linear_fc1 is TELayerNormColumnParallelLinear
try:
    for normalization, fp8, fusion in product(
        ["LayerNorm", "RMSNorm"], [False, True], [False, True]
    ):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            params_dtype=torch.bfloat16,
            bf16=True,
            normalization=normalization,
            tensor_model_parallel_size=world_size,
            sequence_parallel=world_size > 1,
            layernorm_zero_centered_gamma=True,
            gradient_accumulation_fusion=fusion,
        )
        with te.fp8_model_init(enabled=fp8):
            module = TELayerNormColumnParallelLinear(
                128,
                256,
                config=config,
                init_method=torch.nn.init.normal_,
                gather_output=False,
                bias=False,
                skip_bias_add=True,
                is_expert=False,
                tp_comm_buffer_name="qkv",
            )
        assert is_column_parallel_linear(module)
        if normalization == "RMSNorm":
            assert isinstance(module, te.LayerNormLinear)
            # RMSNorm construction must return the original fused module; the
            # type check proves it.  (A plain _tma_fallback
            # getattr is NOT a valid check here: with the native-unfused TE
            # patch active, megatron's real class subclasses the fallback
            # class and inherits the marker as a class attribute.)
            assert (
                type(module).__module__ == "megatron.core.extensions.transformer_engine"
            )
        else:
            # MT-TE 2.0.0 class hierarchy: LayerNormLinear derives from
            # TransformerEngineBaseModule, not te.Linear (the old smoke's
            # isinstance(module, te.Linear) was written for an older TE).
            # The fallback registers as a (virtual) LayerNormLinear subclass.
            assert isinstance(module, te.LayerNormLinear)
            assert module._tma_fallback
        if fusion:
            module.weight.main_grad = torch.zeros(
                module.weight.shape, device=module.weight.device, dtype=torch.float32
            )
        x = torch.randn(
            16, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        with te.fp8_autocast(
            enabled=fp8, fp8_recipe=DelayedScaling(fp8_format=Format.HYBRID)
        ):
            y, bias = module(x)
            loss = y.float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        for p in [x, module.layer_norm_weight]:
            assert p.grad is not None and torch.isfinite(p.grad).all()
        weight_grad = module.weight.main_grad if fusion else module.weight.grad
        assert weight_grad is not None and torch.isfinite(weight_grad).all()
        sd = module.sharded_state_dict(
            metadata={
                "dp_cp_group": parallel_state.get_data_parallel_group(
                    with_context_parallel=True
                )
            }
        )
        assert sd["layer_norm_weight"].global_shape == (128,)
        assert sd["layer_norm_weight"].axis_fragmentations == (1,)
        print(
            "TE_PASS",
            normalization,
            "FP8",
            fp8,
            "fusion",
            fusion,
            "loss",
            loss.item(),
            "keys",
            list(sd),
            flush=True,
        )
finally:
    parallel_state.destroy_model_parallel()
    torch.distributed.destroy_process_group()
