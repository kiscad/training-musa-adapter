"""Module-local control collectives; never replace process-wide torch APIs.

The upstream functions keep their code and module globals. A forwarding torch
namespace intercepts only their control-plane calls, preserving training tensor
collectives and the original startup/checkpoint policy.
"""

from __future__ import annotations

import sys
from contextvars import ContextVar
from datetime import timedelta
from functools import wraps

from ..._engine import AttrPatch

__all__ = ["PATCHES"]

_UNSET = object()


class _DistributedProxy:
    def __init__(self, torch, module_name):
        self._torch = torch
        self._dist = torch.distributed
        self._module_name = module_name
        self._world = None
        self._group = None
        self._active_group = ContextVar("checkpoint_barrier_group", default=_UNSET)

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def checkpoint_group(self, timeout_minutes):
        dist = self._dist
        if not dist.is_initialized() or "mccl" not in str(dist.get_backend()).lower():
            return None
        if self._world is not dist.group.WORLD:
            group = dist.new_group(
                backend="gloo", timeout=timedelta(minutes=timeout_minutes)
            )
            self._world, self._group = dist.group.WORLD, group
        return self._group

    def all_reduce(self, tensor, *args, **kwargs):
        caller = sys._getframe(1)
        torch, dist = self._torch, self._dist
        op = args[0] if args else kwargs.get("op", dist.ReduceOp.SUM)
        group = args[1] if len(args) > 1 else kwargs.get("group")
        asynchronous = args[2] if len(args) > 2 else kwargs.get("async_op", False)
        startup = (
            self._module_name == "megatron.training.training"
            and caller.f_globals.get("__name__") == self._module_name
            and caller.f_code.co_name == "pretrain"
        )
        if (
            startup
            and op == dist.ReduceOp.MIN
            and group is None
            and not asynchronous
            and tensor.dtype == torch.float64
            and tensor.numel() == 1
            and dist.is_initialized()
            and "mccl" in str(dist.get_backend()).lower()
        ):
            # Match synchronize_start_time: truncate to integer microseconds,
            # reduce integers, then restore the original seconds-valued tensor.
            value = torch.tensor(
                [int(tensor.item() * 1_000_000)],
                dtype=torch.int64,
                device=tensor.device,
            )
            result = dist.all_reduce(value, *args, **kwargs)
            tensor.fill_(value.item() / 1_000_000)
            return result
        return dist.all_reduce(tensor, *args, **kwargs)

    def barrier(self, *args, **kwargs):
        caller = sys._getframe(1)
        group = self._active_group.get()
        saving = (
            self._module_name == "megatron.training.checkpointing"
            and caller.f_globals.get("__name__") == self._module_name
            and caller.f_code.co_name == "save_checkpoint"
        )
        if saving and group is not _UNSET and not args and kwargs.get("group") is None:
            kwargs = dict(kwargs, group=group)
        return self._dist.barrier(*args, **kwargs)


class _TorchProxy:
    def __init__(self, torch, module_name):
        self._torch = torch
        self.distributed = _DistributedProxy(torch, module_name)

    def __getattr__(self, name):
        return getattr(self._torch, name)


def _startup_torch(original):
    return _TorchProxy(original, "megatron.training.training")


def _checkpoint_torch(original):
    return _TorchProxy(original, "megatron.training.checkpointing")


def _checkpoint_save(original):
    @wraps(original)
    def save_checkpoint(*args, **kwargs):
        from megatron.training import checkpointing

        torch = checkpointing.torch
        # Independent ONLY/DISABLE selection must not leave half an active fix.
        if not isinstance(torch, _TorchProxy):
            return original(*args, **kwargs)
        dist = torch.distributed
        group = dist.checkpoint_group(
            getattr(checkpointing.get_args(), "distributed_timeout_minutes", 10)
        )
        # Create the host group collectively before any rank enters disk I/O.
        token = dist._active_group.set(group)
        try:
            return original(*args, **kwargs)
        finally:
            dist._active_group.reset(token)

    return save_checkpoint


def _signal_globals(original):
    from signal import Signals

    aliases = [
        (member, f"signal.{name}") for name, member in Signals.__members__.items()
    ]
    return list(original) + [alias for alias in aliases if alias not in original]


PATCHES = (
    AttrPatch(
        id="megatron.training.start-time.integer-microseconds",
        rebind_prefixes=("megatron",),
        target="megatron.training.training:torch",
        # pretrain's float64 MIN startup-timestamp all_reduce exists from
        # core_v0.6.0 (the current megatron package layout) and is unchanged
        # through core_v0.19.0, the newest release line in the checkout.
        version_gates=("megatron-core >=0.6,<0.20",),
        replace=_startup_torch,
        rationale="MCCL 2.11.4 reduced startup float64 MIN timestamps to 1.0.",
        strategy=(
            "Use a module-local torch proxy only for synchronous scalar float64 MIN "
            "calls directly in pretrain; communicate integer microseconds and restore seconds."
        ),
        upstream="NVIDIA/Megatron-LM megatron/training/training.py pretrain",
        remove_when=(
            "Remove when MCCL float64 MIN preserves Unix timestamps and multi-rank "
            "startup timestamps and duration-based exit tests pass."
        ),
    ),
    AttrPatch(
        id="megatron.training.checkpoint.host-barrier-proxy",
        rebind_prefixes=("megatron",),
        target="megatron.training.checkpointing:torch",
        # megatron/training/checkpointing.py (save_checkpoint's barriers)
        # exists from core_v0.6.0; the barriers still run through the module
        # torch global at core_v0.19.0, the newest release line in the
        # checkout.
        version_gates=("megatron-core >=0.6,<0.20",),
        replace=_checkpoint_torch,
        rationale="MCCL device barriers can exceed the device watchdog during slow checkpoint I/O.",
        strategy=(
            "Route only default barriers directly inside save_checkpoint through its "
            "active host group; preserve explicit groups and other callers."
        ),
        upstream="NVIDIA/Megatron-LM megatron/training/checkpointing.py save_checkpoint",
        remove_when=(
            "Remove with host-barrier-context when slow multi-rank checkpoint saves "
            "complete without device watchdog timeouts on the upgraded backend."
        ),
    ),
    AttrPatch(
        id="megatron.training.checkpoint.host-barrier-context",
        rebind_prefixes=("megatron",),
        requires=("megatron.training.checkpoint.host-barrier-proxy",),
        target="megatron.training.checkpointing:save_checkpoint",
        # Same module envelope as host-barrier-proxy; verified through
        # core_v0.19.0, the newest release line in the checkout.
        version_gates=("megatron-core >=0.6,<0.20",),
        replace=_checkpoint_save,
        rationale="All ranks must create the checkpoint host group before diverging into disk I/O.",
        strategy=(
            "Create a world Gloo group before saving, cache by world identity, and set "
            "a scoped barrier context restored even on failure; other backends keep default groups."
        ),
        upstream="NVIDIA/Megatron-LM megatron/training/checkpointing.py save_checkpoint",
        remove_when=(
            "Remove together with host-barrier-proxy after slow-filesystem checkpoint "
            "and process-group reinitialization tests pass without this workaround."
        ),
    ),
    AttrPatch(
        id="megatron.serialization.signal-member-globals",
        rebind_prefixes=("megatron",),
        target="megatron.core.safe_globals:SAFE_GLOBALS",
        # megatron/core/safe_globals.py exists from core_v0.14.0;
        # SAFE_GLOBALS is still the list register_safe_globals iterates at
        # core_v0.19.0, the newest release line in the checkout.
        version_gates=("megatron-core >=0.14,<0.20",),
        replace=_signal_globals,
        rationale="Python 3.10 pickles Signals by signal.SIGTERM names, not only the Signals class.",
        strategy=(
            "Extend the list consumed by upstream register_safe_globals with explicit "
            "signal member aliases, retaining weights-only checkpoint loading."
        ),
        upstream="NVIDIA/Megatron-LM megatron/core/safe_globals.py",
        remove_when=(
            "Remove when upstream registers signal member aliases and weights-only "
            "training-argument checkpoint roundtrips pass on supported Python versions."
        ),
    ),
)
