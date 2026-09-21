"""Training compatibility policies, not a blanket ban on fusion or overlap.
Migrated from megatron-musa-patch ``patches/_training.py`` (rev a1090de).
Retired per-patch env switches map to ONLY/DISABLE on the patch IDs.


The former ``megatron.training.ckpt-format.no-torch-dist`` patch is retired:
``torch`` is Megatron's legacy checkpoint format, not an equivalent backend for
``torch_dist``. Rewriting it silently changes save/load semantics and breaks
async-save and FSDP configurations. Select a checkpoint format explicitly;
the bucket-writer workaround lives separately in ``_checkpointing``.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from ..._engine import AttrPatch

__all__ = ["PATCHES"]

logger = logging.getLogger("training_musa_adaptor")


def _enable_pytorch_profile_validate_args(original: Any) -> Any:
    """Select PyTorch profiling instead of the CUDA runtime/NVTX capture path."""

    @functools.wraps(original)
    def validate_args(*args, **kwargs):
        # Normalize the validated result so defaults and replacement namespaces
        # work too; upstream validation errors still propagate unchanged.
        validated = original(*args, **kwargs)
        if getattr(validated, "profile", False) and not getattr(
            validated, "use_pytorch_profiler", False
        ):
            validated.use_pytorch_profiler = True
            if getattr(validated, "rank", 0) == 0:
                logger.warning(
                    "megatron-musa-patch: --profile automatically enables "
                    "--use-pytorch-profiler on MUSA. The configured CUDA "
                    "compatibility runtime does not provide Megatron's "
                    "cudaProfilerStart/cudaProfilerStop and NVTX capture path. "
                    "Profiling ranks and step range are preserved."
                )
        return validated

    return validate_args


_OVERLAP_FLAGS = (
    "overlap_grad_reduce",
    "overlap_param_gather",
    "overlap_param_gather_with_optimizer_step",
)


def _ignore_overlap_flags_validate_args(original: Any) -> Any:
    """Conservative DP-overlap policy for the observed fused-wgrad integration.

    Megatron's DDP backward hook requires a non-None ``param.grad`` when overlap
    is enabled. Some MUSA TransformerEngine fused-wgrad paths used during
    bring-up violated that contract. This does not establish that MCCL cannot
    overlap communication, and these flags are DP flags, not ``tp_comm_overlap``.

    ``DP_OVERLAP=1`` keeps upstream behavior. ``TP_OVERLAP`` remains a legacy
    spelling when ``DP_OVERLAP`` is unset. Re-test the actual TE/DDP combination
    before lifting this default; disabling overlap may reduce throughput.
    """
    if False:
        return None

    @functools.wraps(original)
    def validate_args(args, *positional, **kwargs):
        # Apply before upstream checks dependent options. False also prevents
        # upstream's defaults (which only fill None) from re-enabling overlap.
        defaults = positional[0] if positional else kwargs.get("defaults", {})
        requested = [
            name
            for name in _OVERLAP_FLAGS
            if getattr(args, name, None)
            or (getattr(args, name, None) is None and defaults.get(name, False))
        ]
        for name in _OVERLAP_FLAGS:
            setattr(args, name, False)
        if requested and getattr(args, "rank", 0) == 0:
            logger.warning(
                "megatron-musa-patch: disabling DP overlap options %s for the "
                "fused-wgrad/DDP compatibility policy; this may reduce throughput. "
                "Set MEGATRON_MUSA_PATCH_DP_OVERLAP=1 to test upstream overlap "
                "with your MUSA/TransformerEngine stack.",
                ", ".join(requested),
            )
        return original(args, *positional, **kwargs)

    return validate_args


def _noop_fused_kernels_load(original: Any) -> Any:
    """Skip the legacy nvcc probe/build entry point, not all fused kernels.

    Local Megatron's ``load`` probes ``nvcc -V`` and defines an extension-build
    helper without invoking it. Older releases may also build kernels here;
    consumers such as fused softmax still need a working backend or explicit
    upstream fallback. The dataset index builder is left untouched.
    """

    @functools.wraps(original)
    def load(args=None):
        return None

    return load


def _noop_set_jit_fusion_options(original: Any) -> Any:
    """Skip startup fusion configuration/warmup, retaining runtime fusion code.

    Upstream allocates CUDA tensors and warms bias/GELU, SwiGLU and dropout/add.
    On recent PyTorch it uses torch.compile, not only the old NVIDIA JIT fuser.
    Keep this bring-up policy until the configured MUSA compiler paths have
    passed warmup and training checks; it can affect compilation latency and
    performance. ``JIT_WARMUP=1`` restores upstream behavior.
    """
    if False:
        return None

    @functools.wraps(original)
    def set_jit_fusion_options():
        return None

    return set_jit_fusion_options


PATCHES = (
    AttrPatch(
        id="megatron.training.profile.pytorch",
        rebind_prefixes=("megatron",),
        target="megatron.training.arguments:validate_args",
        replace=_enable_pytorch_profile_validate_args,
        rationale=(
            "Bare --profile enters cudaProfilerStart/Stop and emit_nvtx in "
            "training.py; the configured MUSA CUDA-compatibility runtime does "
            "not supply that capture path."
        ),
        strategy=(
            "After argument validation enable use_pytorch_profiler with a rank-zero "
            "warning, preserving the user's profile ranks and step range."
        ),
        upstream="NVIDIA/Megatron-LM megatron/training/training.py; arguments.py",
        remove_when=(
            "Review on torch_musa/torchada and Megatron upgrades; remove when "
            "Megatron selects a backend-compatible profiler or the original "
            "runtime/NVTX capture path passes rank and step-range smoke tests."
        ),
    ),
    AttrPatch(
        id="megatron.legacy.fused-kernels.load.noop",
        rebind_prefixes=("megatron",),
        target="megatron.legacy.fused_kernels:load",
        replace=_noop_fused_kernels_load,
        rationale=(
            "_compile_dependencies invokes a legacy loader that probes nvcc and "
            "defines a CUDA extension builder; this is not a MUSA build route."
        ),
        strategy=(
            "No-op only this loader, leaving dataset compilation and other "
            "fusion/kernel paths intact. This does not provide missing softmax kernels."
        ),
        upstream="NVIDIA/Megatron-LM megatron/legacy/fused_kernels/__init__.py",
        remove_when=(
            "Review loader and kernel consumers on every Megatron upgrade; remove "
            "when the loader is no longer invoked or selects a validated MUSA "
            "toolchain. Re-test any legacy fused-softmax configuration separately."
        ),
    ),
    AttrPatch(
        id="megatron.training.set-jit-fusion-options.noop",
        rebind_prefixes=("megatron",),
        target="megatron.training.training:set_jit_fusion_options",
        replace=_noop_set_jit_fusion_options,
        rationale=(
            "training.py imports initialize.py's startup helper, which creates "
            "CUDA warmup tensors and exercises compiler/fusion paths that need "
            "validation on the configured MUSA stack."
        ),
        strategy=(
            "Skip the startup helper by default without replacing runtime fusion "
            "operators; JIT_WARMUP=1 retains upstream setup and warmup."
        ),
        upstream="NVIDIA/Megatron-LM megatron/training/training.py; initialize.py",
        remove_when=(
            "Review on PyTorch, torch_musa and Megatron upgrades; remove after "
            "JIT_WARMUP=1 passes bias/GELU, SwiGLU, dropout forward/backward and "
            "representative training/compile-latency tests."
        ),
    ),
    AttrPatch(
        id="megatron.training.initialize.set-jit-fusion-options.noop",
        rebind_prefixes=("megatron",),
        target="megatron.training.initialize:set_jit_fusion_options",
        replace=_noop_set_jit_fusion_options,
        rationale=(
            "initialize.py defines the same CUDA-oriented startup helper that "
            "training.py imports; direct initialize callers need the same policy."
        ),
        strategy=(
            "Apply the same opt-out warmup replacement at the defining module, "
            "covering direct calls as well as training.py's alias."
        ),
        upstream="NVIDIA/Megatron-LM megatron/training/initialize.py",
        remove_when=(
            "Remove together with the training.set-jit-fusion-options patch after "
            "the configured compiler/fusion stack passes JIT_WARMUP=1 training tests."
        ),
    ),
    AttrPatch(
        id="megatron.training.overlap-flags.noop",
        rebind_prefixes=("megatron",),
        target="megatron.training.arguments:validate_args",
        replace=_ignore_overlap_flags_validate_args,
        rationale=(
            "The observed MUSA TE fused-wgrad integration can leave param.grad "
            "None, violating Megatron DDP's overlap_grad_reduce backward-hook "
            "assertion. This is an integration contract issue, not evidence "
            "that all MCCL stream overlap is unsupported."
        ),
        strategy=(
            "Before validation disable DP gradient reduction, parameter gather "
            "and its optimizer-step overlap dependency, warning on changes. "
            "DP_OVERLAP=1 (legacy TP_OVERLAP=1) preserves upstream behavior; "
            "tp_comm_overlap and unrelated communication flags are untouched."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/core/distributed/distributed_data_parallel.py "
            "_make_backward_post_hook; megatron/training/arguments.py"
        ),
        remove_when=(
            "Review on MT-TransformerEngine, torch_musa/MCCL and Megatron upgrades; "
            "remove after DP_OVERLAP=1 passes multi-rank fused-wgrad gradient, "
            "optimizer-state and convergence parity tests plus an overlap benchmark."
        ),
    ),
)
