"""Best-effort process-group teardown and collective-shape adapters.
Migrated from megatron-musa-patch ``patches/_distributed.py`` (rev a1090de).
Retired per-patch env switches map to ONLY/DISABLE on the patch IDs.


Runs that leave MCCL groups live during Python finalization have exhibited
watchdog errors and, on some stacks, aborts. An atexit callback improves the
normal-exit path; it cannot make teardown deterministic across ranks, handle
SIGKILL, or replace a launcher's explicit coordinated cleanup.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable

from ..._engine import AttrPatch, HookPatch
from ...backends import musa_available as _musa_live

__all__ = ["PATCHES"]

logger = logging.getLogger("training_musa_adaptor")

_teardown_callback: Callable[[], None] | None = None


def _install_clean_teardown() -> bool | None:
    global _teardown_callback
    if _teardown_callback is not None:
        return False

    import atexit

    import torch.distributed as dist

    def _teardown() -> None:
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:  # noqa: BLE001 - best effort during interpreter exit
            pass

    atexit.register(_teardown)
    _teardown_callback = _teardown


def _uninstall_clean_teardown() -> None:
    global _teardown_callback
    if _teardown_callback is None:
        return
    import atexit

    atexit.unregister(_teardown_callback)
    _teardown_callback = None


def _fsdp_gradient_reduce_prescale(original: Any) -> Any:
    """Prescale gradients on device and reduce with SUM instead of PREMUL_SUM.

    torch_musa's ``ProcessGroupMCCL`` has no ``PreMulSum`` implementation: the
    op Megatron builds with ``torch.distributed._make_nccl_premul_sum`` reaches
    MCCL as an unmapped enum value and every FSDP gradient reduction fails with
    ``RuntimeError: Unexpected ReduceOp: \\x08``. Multiplying the buffer in
    place on the same stream keeps the reduction math (and its rounding) on the
    device and preserves the async collective contract.
    """
    import torch

    @functools.wraps(original)
    def gradient_reduce_preprocessing(grad_data, scaling_factor, ddp_config):
        if (
            scaling_factor is not None
            and not ddp_config.average_in_collective
            and ddp_config.gradient_reduce_div_fusion
            and grad_data.dtype != torch.bfloat16
        ):
            # Exactly the branch that builds a PREMUL_SUM op upstream. Scale in
            # place on the current stream -- the caller issues its collective on
            # this buffer next -- and reduce with SUM.
            logger.debug(
                "FSDP gradient prescale: dtype=%s factor=%r (PREMUL_SUM unavailable)",
                grad_data.dtype,
                scaling_factor,
            )
            grad_data.mul_(scaling_factor)
            return torch.distributed.ReduceOp.SUM
        return original(grad_data, scaling_factor, ddp_config)

    return gradient_reduce_preprocessing


class _SubgroupsDistributedProxy:
    """Forwards ``torch.distributed`` for callers that create subgroups.

    torchada already translates ``init_process_group`` and ``new_group``, but
    ``new_subgroups_by_enumeration`` resolves ``new_group`` as a global inside
    ``torch.distributed.distributed_c10d``, bypassing those wrappers. Megatron's
    bridge communicator and hyper_comm_grid therefore still hand an explicit
    ``backend='nccl'`` to process-group creation, which dies with "Distributed
    package doesn't have NCCL built in". This proxy translates exactly that
    request and forwards everything else untouched.
    """

    def __init__(self, dist):
        self._dist = dist

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def new_subgroups_by_enumeration(self, *args, **kwargs):
        backend = kwargs.get("backend")
        positional = None
        if backend is None and len(args) >= 3:  # (ranks, timeout, backend, ...)
            positional = 2
            backend = args[2]
        if isinstance(backend, str) and backend.lower() == "nccl" and _musa_live():
            backend = "mccl"
            if positional is not None:
                args = args[:positional] + (backend,) + args[positional + 1 :]
            else:
                kwargs["backend"] = backend
        return self._dist.new_subgroups_by_enumeration(*args, **kwargs)


def _subgroups_distributed_proxy(original: Any) -> Any:
    return _SubgroupsDistributedProxy(original)


PATCHES = (
    HookPatch(
        id="torch.distributed.clean-teardown",
        trigger="megatron.core.parallel_state",
        run=_install_clean_teardown,
        undo=_uninstall_clean_teardown,
        rationale=(
            "MUSA bring-up runs that left process groups alive at interpreter exit "
            "showed MCCL watchdog/finalization errors. The affected entry points "
            "did not explicitly destroy the default group."
        ),
        strategy=(
            "Register one best-effort atexit callback to destroy an initialized "
            "process group, without adding a barrier. TEARDOWN=0 skips registration; "
            "undo unregisters only this callback and does not destroy a live group."
        ),
        upstream="pytorch/pytorch torch/distributed/distributed_c10d.py; MCCL shutdown",
        remove_when=(
            "Review on torch_musa/MCCL and launcher/Megatron upgrades; remove when "
            "the entry point owns explicit cleanup or repeated multi-rank normal "
            "exit tests pass with TEARDOWN=0. Test failure/interrupt paths separately."
        ),
    ),
    AttrPatch(
        id="megatron.fsdp.premul-sum.device-prescale",
        rebind_prefixes=("megatron",),
        target=(
            "megatron.core.distributed.fsdp.src.megatron_fsdp.param_and_grad_buffer:"
            "gradient_reduce_preprocessing"
        ),
        replace=_fsdp_gradient_reduce_prescale,
        rationale=(
            "FSDP gradient averaging with gradient_reduce_div_fusion builds a "
            "PREMUL_SUM reduce op via torch.distributed._make_nccl_premul_sum. "
            "torch_musa's ProcessGroupMCCL does not implement PreMulSum, so the op "
            "reaches MCCL as an unmapped enum value and every gradient reduction "
            "fails with 'RuntimeError: Unexpected ReduceOp: \\x08' before any "
            "training step. Observed on torch_musa 2.7.1 with Megatron "
            "core_v0.16.1 in distributed/megatron_fsdp tests."
        ),
        strategy=(
            "Intercept exactly the PREMUL_SUM branch (scaling_factor is not None, "
            "average_in_collective off, gradient_reduce_div_fusion on, dtype not "
            "bf16): multiply the gradient buffer in place on the current stream -- "
            "the caller issues its reduce-scatter/all-reduce on the same buffer "
            "next -- and return ReduceOp.SUM. Every other branch, including AVG "
            "and the bf16 path, calls the original function unchanged, so no "
            "scaling is applied twice and the async Work contract is untouched."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/core/distributed/fsdp/src/megatron_fsdp/"
            "param_and_grad_buffer.py:gradient_reduce_preprocessing"
        ),
        remove_when=(
            "Remove when torch_musa/MCCL supports PreMulSum: disable this patch id "
            "and re-run the megatron_fsdp unit tests plus one multi-rank training "
            "step with gradient_reduce_div_fusion enabled; delete only if the "
            "unpatched path passes."
        ),
    ),
    AttrPatch(
        id="megatron.bridge-communicator.subgroups-backend",
        rebind_prefixes=("megatron",),
        target="megatron.core.pipeline_parallel.bridge_communicator:dist",
        replace=_subgroups_distributed_proxy,
        rationale=(
            "The bridge communicator creates its boundary broadcast subgroups "
            "with dist.new_subgroups_by_enumeration(backend='nccl'). torchada "
            "translates nccl->mccl for init_process_group and new_group, but "
            "new_subgroups_by_enumeration resolves new_group as a global inside "
            "torch.distributed.distributed_c10d and bypasses those wrappers, so "
            "every pipeline-bridge setup fails with 'Distributed package doesn't "
            "have NCCL built in'. Observed on torch 2.7.1/torch_musa 2.7.1 with "
            "Megatron core_v0.16.1."
        ),
        strategy=(
            "Bind the module's torch.distributed global to a forwarding proxy "
            "that translates only an exact 'nccl' backend string (Backend.NCCL "
            "is a str subclass, so it is covered) to 'mccl' while a live MUSA "
            "runtime is present, whether it arrives positionally or by keyword. "
            "None, gloo, mccl and every other value pass through untouched, as "
            "do ranks enumeration, timeout, pg_options, group_desc and the "
            "(current_subgroup, subgroups) return contract."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/core/pipeline_parallel/"
            "bridge_communicator.py; pytorch torch/distributed/"
            "distributed_c10d.py:new_subgroups_by_enumeration"
        ),
        remove_when=(
            "Remove when torchada or c10d translates this entry natively: disable "
            "this patch id and re-run pipeline_parallel/test_bridge_communicator "
            "on MUSA; delete only if the unpatched path passes."
        ),
    ),
    AttrPatch(
        id="megatron.hyper-comm-grid.subgroups-backend",
        rebind_prefixes=("megatron",),
        target="megatron.core.hyper_comm_grid:dist",
        replace=_subgroups_distributed_proxy,
        rationale=(
            "HyperCommGrid creates its process groups with "
            "dist.new_subgroups_by_enumeration(rank_enum, backend=self.backend) "
            "and the same c10d-internal new_group bypass as the bridge "
            "communicator, so an 'nccl' grid backend fails identically."
        ),
        strategy=(
            "Bind the module's torch.distributed global to the same forwarding "
            "proxy used for the bridge communicator; only the exact 'nccl' "
            "request is translated and every other attribute forwards to the "
            "real namespace."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/hyper_comm_grid.py",
        remove_when=(
            "Remove together with the bridge-communicator patch when torchada or "
            "c10d translates this entry natively; re-run test_hyper_comm_grid "
            "subgroup cases with the patch disabled before deleting."
        ),
    ),
)
