"""Repair DCP CPU staging and avoid Megatron bucket-worker forks.
Migrated from megatron-musa-patch ``patches/_checkpointing.py`` (rev a1090de).
Retired per-patch env switches map to ONLY/DISABLE on the patch IDs.


DCP's CUDA availability probe sees the emulated API rather than the tensor's
actual MUSA device. Correct its local device selection so the existing loader
performs and synchronizes the device-to-host copy before serialization.

The bring-up stack exhibited a child-process torch.save segfault followed by
``count_queue.join()`` hanging after MUSA initialization. Local upstream forks
bucket workers with ``mp.get_context('fork')``; inherited runtime state is a
fork-safety concern, not proof that all distributed checkpointing is unusable.

Upstream preloads tensors to CPU before invoking this writer. Reuse its bucket
serialization synchronously in the calling process, trading away bucket-level
parallelism. This does NOT remove forks in the external async caller (see
``async_utils.DynamicAsyncCaller``), change the checkpoint format, or establish
end-to-end async save safety. Review those paths separately on stack upgrades.
"""

from __future__ import annotations

from collections import deque
from functools import wraps
from typing import Any

from ..._engine import AttrPatch, HookPatch

__all__ = ["PATCHES"]

_dcp_device_owned: tuple[Any, Any, Any] | None = None


def _install_dcp_device() -> bool:
    """Correct only DCP's imported device selector, after Megatron activation."""
    global _dcp_device_owned
    import importlib
    import sys

    torch = sys.modules.get("torch")
    musa = getattr(torch, "musa", None)
    if _dcp_device_owned is not None or torch is None or musa is None or not musa.is_available():
        return False
    filesystem = importlib.import_module("torch.distributed.checkpoint.filesystem")
    original = getattr(filesystem, "_get_available_device_type", None)
    if original is None:
        return False

    @wraps(original)
    def actual_device_type():
        device_type = original()
        # CUDA API availability is emulated by torchada, but tensor.device.type
        # remains 'musa'. DCP compares these strings before staging to CPU.
        # Inspect the actual stream at call time, independent of hook ordering.
        if device_type == "cuda" and torch.cuda.current_stream().device.type == "musa":
            return "musa"
        return device_type

    filesystem._get_available_device_type = actual_device_type  # type: ignore[attr-defined]
    _dcp_device_owned = (filesystem, original, actual_device_type)
    return True


def _uninstall_dcp_device() -> None:
    global _dcp_device_owned
    if _dcp_device_owned is not None:
        module, original, replacement = _dcp_device_owned
        if getattr(module, "_get_available_device_type", None) is replacement:
            module._get_available_device_type = original
        _dcp_device_owned = None


class _ImmediateQueue:
    """Minimal stand-in for ``mp.SimpleQueue`` (put once, get once)."""

    def __init__(self) -> None:
        self._items: deque[Any] = deque()

    def put(self, item: Any) -> None:
        self._items.append(item)

    def get(self) -> Any:
        if not self._items:
            raise RuntimeError("megatron-musa-patch: result queue is empty")
        return self._items.popleft()


class _Counter:
    """Minimal stand-in for ``mp.JoinableQueue``'s get/task_done handshake."""

    def __init__(self) -> None:
        self._count = 0

    def put(self, _item: Any = None) -> None:
        self._count += 1

    def get(self) -> bool:
        if self._count <= 0:
            raise RuntimeError("megatron-musa-patch: count queue underflow")
        self._count -= 1
        return True

    def task_done(self) -> None:
        return None


def _serial_writer(original: Any) -> Any:
    """Sequential replacement for ``write_preloaded_data_multiproc``."""
    @wraps(original)
    def write_preloaded_data_serial(
        transform_list, use_msc, rank, write_buckets, global_results_queue
    ):
        # Local import: this module must stay importable without torch.
        import logging

        from megatron.core.dist_checkpointing.strategies.filesystem_async import (
            FileSystemWriterAsync,
        )

        logger = logging.getLogger(__name__)
        logger.debug(
            "megatron-musa-patch: writing %d checkpoint bucket(s) without forking",
            len(write_buckets),
        )

        # Resolve at call time to preserve any wrapper installed on the worker.
        # A rename/signature change requires review; this adapter deliberately
        # relies on upstream's CPU staging and per-bucket result protocol.
        write_one_bucket = FileSystemWriterAsync.write_preloaded_data

        results_queue = _ImmediateQueue()
        count_queue = _Counter()
        # Same payload contract as upstream's forked writer: a dict of
        # per-bucket results, or a single Exception.  Upstream's
        # ``retrieve_write_results`` raises "Worker failure" only when the
        # payload *is* an Exception, so a failed bucket must replace the
        # payload -- putting it into the dict would sail through both of the
        # consumer's checks (isinstance / length) and explode later.
        write_results_or_exc: dict | Exception = {}

        for local_proc_idx, write_bucket in enumerate(write_buckets):
            count_queue.put(local_proc_idx)
            try:
                write_one_bucket(
                    transform_list,
                    local_proc_idx,
                    write_bucket,
                    results_queue,
                    count_queue,
                    use_fsync=True,
                    use_msc=use_msc,
                )
                # Missing/malformed output is also a worker failure. Publish an
                # Exception instead of letting a protocol error escape before
                # the caller receives anything on its global result queue.
                idx, result = results_queue.get()
                if idx != local_proc_idx:
                    raise RuntimeError(
                        f"checkpoint bucket {local_proc_idx} returned unexpected index {idx!r}"
                    )
                if isinstance(result, Exception):
                    raise result
                if not isinstance(result, list):
                    raise TypeError(
                        f"checkpoint bucket {idx} returned {type(result).__name__}, not list"
                    )
                write_results_or_exc[idx] = result  # type: ignore[index]
            except Exception as exc:  # noqa: BLE001 - report upstream's failure payload
                logger.error("megatron-musa-patch: bucket %d failed: %s", local_proc_idx, exc)
                write_results_or_exc = exc
                break

        global_results_queue.put(write_results_or_exc)

    # Upstream declares this as a ``staticmethod``; keep that so that
    # ``self.write_preloaded_data_multiproc`` does not bind ``self`` and shift
    # the argument list (upstream builds a ``functools.partial`` from it).
    return staticmethod(write_preloaded_data_serial)


PATCHES = (
    HookPatch(
        id="megatron.dist-ckpt.musa-cpu-staging",
        trigger="megatron.core.parallel_state",
        run=_install_dcp_device,
        undo=_uninstall_dcp_device,
        rationale=(
            "With CUDA API emulation, torch 2.7 DCP selects cuda while tensor devices "
            "remain musa. _OverlappingCpuLoader skips the CPU copy and the writer "
            "fails assert tensor.is_cpu, including FSDP DTensor checkpoints."
        ),
        strategy=(
            "Own only filesystem's imported device selector. When its cuda selection "
            "resolves to an actual MUSA stream, return musa. Reuse upstream staging, "
            "stream synchronization, planners and serialization without changing "
            "checkpoint keys or sharding. No additional synchronous-copy fallback."
        ),
        upstream="PyTorch torch/distributed/checkpoint/filesystem.py:_OverlappingCpuLoader",
        remove_when=(
            "Remove when disabling this hook passes tiny multi-rank DTensor model/optimizer "
            "save/load and original FSDP checkpoint tests under CUDA API emulation. "
            "Review the private filesystem selector on PyTorch upgrades."
        ),
    ),
    AttrPatch(
        id="megatron.dist-ckpt.no-fork-writer",
        rebind_prefixes=("megatron",),
        target=(
            "megatron.core.dist_checkpointing.strategies.filesystem_async:"
            "FileSystemWriterAsync.write_preloaded_data_multiproc"
        ),
        replace=_serial_writer,
        rationale=(
            "The affected MUSA stack showed torch.save segfaulting in a forked "
            "bucket worker, leaving the parent in count_queue.join(). Upstream "
            "forks those workers after device initialization despite already "
            "staging their tensor data on CPU."
        ),
        strategy=(
            "Run upstream's per-bucket writer sequentially in the calling "
            "process, preserving its dict-or-Exception result protocol and "
            "staticmethod binding. This sacrifices bucket I/O parallelism, "
            "not checkpoint sharding, and does not remove external async forks. "
            "CKPT_FORK=1 restores the original worker launcher."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/core/dist_checkpointing/strategies/" "filesystem_async.py"
        ),
        remove_when=(
            "Review worker signatures/result protocol and async_utils on every "
            "Megatron/PyTorch/torch_musa upgrade. Remove after upstream uses a "
            "validated safe launcher or CKPT_FORK=1 passes multi-rank post-init "
            "save/load and worker-failure tests without hangs; benchmark writer "
            "throughput separately. Spawn alone is not a sufficient safety test."
        ),
    ),
)
