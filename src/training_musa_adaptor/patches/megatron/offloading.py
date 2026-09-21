"""Keep resident TE weights out of activation-only offloading.

Migrated from megatron-musa-patch ``patches/_offloading.py`` (rev a1090de).
Retired per-patch env switches map to ONLY/DISABLE on the patch IDs.
"""


from __future__ import annotations

import functools

from ..._engine import AttrPatch
from ...backends import musa_available

__all__ = ["PATCHES"]


def _activation_only(original):
    if not musa_available():
        return None

    @functools.wraps(original)
    def checker(self, tensor):
        import torch

        # MT-TE saves Parameters (and marks detached weight tensors using
        # weight_offloading), whereas this mcore handler only understands
        # offloading_activation. Copying resident weights cannot free them.
        if (
            tensor.device.type == "cpu"
            or isinstance(tensor, torch.nn.Parameter)
            or getattr(tensor, "weight_offloading", False)
        ):
            return False
        return original(self, tensor)

    return checker


def _preserve_offload_markers(original):
    if not musa_available():
        return None

    @functools.wraps(original)
    def prepare(*tensors):
        import torch

        saved, objects = original(*tensors)
        # Quantized tensors expand into multiple storages; their native
        # protocol is intentionally outside this ordinary-tensor adapter.
        if len(saved) == len(tensors) and all(
            t is None or type(t) in (torch.Tensor, torch.nn.Parameter) for t in tensors
        ):
            for source, detached in zip(tensors, saved):
                if source is None or detached is None:
                    continue
                if (
                    isinstance(source, torch.nn.Parameter)
                    or getattr(source, "weight_offloading", False)
                    or getattr(source, "offloading_activation", None) is False
                ):
                    detached.offloading_activation = False
        return saved, objects

    return prepare


PATCHES = (
    AttrPatch(
        id="transformer_engine.saved-tensors.offload-markers",
        target="transformer_engine.pytorch.tensor.quantized_tensor:prepare_for_saving",
        rebind_prefixes=("transformer_engine",),
        version_gates=("transformer_engine >=2.0,<2.1",),
        replace=_preserve_offload_markers,
        rationale="MT-TE prepare_for_saving uses tensor.data, losing Parameter identity and offload markers.",
        strategy=(
            "Propagate the activation-offload exclusion onto detached ordinary saved tensors "
            "for Parameters, weight_offloading tensors and explicit non-offloadable activations. "
            "Keep native storage, order and restoration metadata; quantized protocols stay native."
        ),
        upstream="MT-TE pytorch/tensor/quantized_tensor.py:prepare_for_saving",
        remove_when="Remove when native saved tensors retain activation-offload exclusions and memory tests pass.",
    ),
    AttrPatch(
        id="megatron.offloading.resident-parameters",
        rebind_prefixes=("megatron",),
        target="megatron.core.pipeline_parallel.fine_grained_activation_offload:ChunkOffloadHandler.tensor_need_offloading_checker",
        replace=_activation_only,
        rationale=(
            "MT-TE saves resident Parameters and weight_offloading-tagged tensors inside "
            "activation contexts. Mcore copies and counts them as releasable activations; "
            "expert_fc1 reports 280 MiB while only 56 MiB can be released."
        ),
        strategy=(
            "Exclude Parameters, vendor-tagged weights and CPU tensors from activation-only "
            "offloading. Keep them in the saved-tensor group for backward, retaining original "
            "activation filtering and statistics for tensors actually transferred."
        ),
        upstream="Megatron-LM fine_grained_activation_offload.py:ChunkOffloadHandler; MT-TE Linear/GroupedLinear",
        remove_when="Remove when TE/mcore activation offload excludes resident weights and memory tests pass.",
    ),
)
