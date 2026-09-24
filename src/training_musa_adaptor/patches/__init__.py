"""The patch ledger.

Every patch this package applies lives in one of the modules below as
declarative data.  ``MODULES`` order matters for patches touching the same
symbol (earlier factories build the inner wrappers); the device
compatibility layer goes first so anything reading ``torch.cuda`` while its
own replacement is being built already sees a working namespace.

Adding a patch means adding a record in the matching module -- you should
never need to copy an upstream file into this repository.  Modules imported
here must stay standard-library + this package's lightweight modules at top
level; torch, frameworks and optional accelerator libraries load inside
factories or safe initialization functions.
"""

from __future__ import annotations

__all__ = ["PATCHES", "MODULES", "SUITES", "PATCH_SUITES"]

from . import deepspeed, peft, platform, transformer_engine
from .mcore_bridge import ssm as mcore_bridge_ssm
from .megatron import (
    attention,
    checkpointing,
    control_collectives,
    cuda_graphs,
    delayed_wgrad,
    device_arch,
    distributed,
    grouped_gemm,
    layer_norm,
    moe,
    offloading,
    rope,
    softmax,
    ssm,
    training,
)
from .transformers import rms_norm

MODULES = (
    platform,
    delayed_wgrad,
    device_arch,
    distributed,
    transformer_engine,
    cuda_graphs,
    attention,
    layer_norm,
    moe,
    offloading,
    grouped_gemm,
    rope,
    softmax,
    training,
    checkpointing,
    control_collectives,
    ssm,
    mcore_bridge_ssm,
    rms_norm,
    peft,
    deepspeed,
)

PATCHES = tuple(patch for module in MODULES for patch in module.PATCHES)

#: Adaptation domain a patch belongs to.  Suite names are accepted in
#: ONLY/DISABLE (environment and TOML) and expand to every patch id of that
#: suite, so a whole domain can be toggled at once (e.g. when another team
#: owns the Megatron MUSA adaptation).
SUITES = {
    "platform": (platform,),
    "megatron": (
        delayed_wgrad,
        device_arch,
        distributed,
        cuda_graphs,
        attention,
        layer_norm,
        moe,
        offloading,
        grouped_gemm,
        rope,
        softmax,
        training,
        checkpointing,
        control_collectives,
        ssm,
    ),
    "transformer_engine": (transformer_engine,),
    "transformers": (rms_norm,),
    "peft": (peft,),
    "deepspeed": (deepspeed,),
    "mcore_bridge": (mcore_bridge_ssm,),
}

#: patch id -> suite name (covers every registered patch exactly once).
PATCH_SUITES = {
    patch.id: suite
    for suite, modules in SUITES.items()
    for module in modules
    for patch in module.PATCHES
}
