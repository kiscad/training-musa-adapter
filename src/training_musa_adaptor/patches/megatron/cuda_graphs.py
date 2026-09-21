"""Open Megatron's TE CUDA-graph import gate for the MT-TE fork.
Migrated from megatron-musa-patch ``patches/_cuda_graphs.py`` (rev a1090de).
Retired per-patch env switches map to ONLY/DISABLE on the patch IDs.


``megatron/core/transformer/cuda_graphs.py`` imports its whole Transformer
Engine graph surface inside one ``try`` block, including
``transformer_engine.pytorch.utils.make_weak_ref`` -- a symbol newer NVIDIA TE
releases ship and the installed MT-TE 2.0 fork does not. The bare ``except``
then leaves ``HAVE_TE_GRAPHS = False``: every local-graph entry dies on
``RNG tracker does not support cudagraphs!`` (``CudaGraphManager``) or
``CUDA Graphs are not supported without TE.`` (``TECudaGraphHelper``) before
any capture is attempted. On the r2 sweep that accounted for the whole of
``transformer/test_cuda_graphs.py`` -- 23 failures and 3 errors.

This hook ports ``make_weak_ref`` so the gate opens, and installs it only when
every hard prerequisite of Megatron's graph paths is really present on the
running stack; anything missing declines with the reason in the report. No
graph math, RNG handling or capture logic is implemented here.
"""

from __future__ import annotations

import logging
from typing import Any

from ..._engine import HookPatch

__all__ = ["PATCHES"]

logger = logging.getLogger("training_musa_adaptor")

#: (module, function) this hook owns while applied.
_owned: tuple[Any, Any] | None = None


def _musa_fork_installed() -> bool:
    """Is the installed TransformerEngine the MUSA fork?"""
    import importlib.metadata as md
    from pathlib import Path

    try:
        distribution = md.distribution("transformer_engine")
        musa_dir = Path(str(distribution.locate_file("transformer_engine/musa")))
    except Exception:  # noqa: BLE001 - distribution metadata is best effort
        return False
    return musa_dir.exists()


def _missing_torch_graph_apis(torch: Any) -> list[str]:
    """Hard torch-level requirements of Megatron's graph paths.

    Probes run on the live modules. ``register_generator_state`` on the
    ``CUDAGraph`` binding doubles as the discriminator between a working MUSA
    graph class and the CUDA-spelled stub of an unadapted build. No probe
    captures, allocates graph memory or consumes RNG state.
    """
    missing: list[str] = []

    musa = getattr(torch, "musa", None)
    if musa is None or not getattr(musa, "is_available", lambda: False)():
        missing.append("the MUSA runtime is not available")
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not getattr(cuda, "is_available", lambda: False)():
        missing.append("torch.cuda does not report an available device")
    if missing:
        return missing

    graph_cls = getattr(cuda, "CUDAGraph", None)
    if not (
        isinstance(graph_cls, type)
        and callable(getattr(graph_cls, "register_generator_state", None))
    ):
        missing.append("torch.cuda.CUDAGraph cannot register generator states (graph-safe RNG)")
    for name in (
        "graph",
        "graph_pool_handle",
        "synchronize",
        "current_stream",
        "default_stream",
        "set_stream",
        "Stream",
    ):
        if not callable(getattr(cuda, name, None)):
            missing.append(f"torch.cuda.{name} is missing")
    for name in ("graphsafe_set_state", "graphsafe_get_state", "clone_state"):
        if not callable(getattr(torch.Generator, name, None)):
            missing.append(f"torch.Generator.{name} is missing")
    # The capture context must be routed to the MUSA graph implementation: on
    # an unadapted build torch.cuda.graph is the stock CUDA wrapper with no
    # capture bindings behind it.
    musa_graph_ctx = getattr(musa, "graph", None)
    graph_ctx = getattr(cuda, "graph", None)
    if (
        callable(graph_ctx)
        and graph_ctx is not musa_graph_ctx
        and graph_ctx is getattr(getattr(cuda, "graphs", None), "graph", None)
    ):
        missing.append(
            "torch.cuda.graph is the stock CUDA wrapper; the device adapter must "
            "route it to the MUSA graph context"
        )
    return missing


def _missing_te_graph_apis() -> list[str]:
    """TE-side symbols Megatron's ``cuda_graphs`` try-block consumes."""
    missing: list[str] = []
    try:
        import transformer_engine.pytorch.graph as te_graph
        from transformer_engine.pytorch.distributed import graph_safe_rng_available
    except Exception as exc:  # noqa: BLE001 - report the real import failure
        return [f"the TransformerEngine graph surface is not importable ({exc})"]
    for name in (
        "make_graphed_callables",
        "save_fp8_tensors",
        "restore_fp8_tensors",
        "set_capture_start",
        "set_capture_end",
    ):
        if not callable(getattr(te_graph, name, None)):
            missing.append(f"transformer_engine.pytorch.graph.{name} is missing")
    try:
        if not graph_safe_rng_available():
            missing.append("TransformerEngine reports graph-safe RNG as unavailable")
    except Exception as exc:  # noqa: BLE001
        missing.append(f"graph_safe_rng_available() failed ({exc})")
    return missing


def _missing_graph_prerequisites() -> list[str]:
    """Everything Megatron's graph paths need that this stack must really have."""
    import sys

    torch = sys.modules.get("torch")
    if torch is None:
        return ["torch is not imported yet"]
    missing = _missing_torch_graph_apis(torch)
    return missing if missing else _missing_te_graph_apis()


def _zero_copy_alias(x: Any) -> Any:
    """Zero-copy tensor over the input's memory.

    NVIDIA TE rebuilds a non-owning tensor from the raw data pointer through
    ``__cuda_array_interface__``; torch builds without the CUDA bindings (MUSA)
    cannot create tensors from raw pointers that way -- ``torch.as_tensor``
    fails with "Cannot initialize CUDA without ATen_cuda library" (verified on
    torch_musa 2.11) -- so the alias shares the input's storage instead. Same
    data pointer, shape, dtype and strides, no autograd history. The deviation
    is ownership: the alias keeps the buffer alive until the runner drops it,
    which is the higher-memory behavior Megatron itself documents for runs
    without TE's weak references.
    """
    import torch

    alias = torch.empty(0, dtype=x.dtype, device=x.device)
    alias.set_(x.untyped_storage(), x.storage_offset(), x.shape, x.stride())
    return alias


def _build_make_weak_ref() -> Any:
    """Build the ``make_weak_ref`` port; torch is imported lazily."""
    import torch

    def make_weak_ref(x: Any) -> Any:
        """Port of NVIDIA TransformerEngine's ``make_weak_ref``.

        Returns containers recursively; CUDA/MUSA tensors become zero-copy
        aliases of the same memory so graph I/O buffers do not pin Python
        references beyond what Megatron's reuse pool intends. CPU tensors and
        scalars pass through unchanged, exactly like the upstream function.
        """
        if isinstance(x, torch.Tensor):
            if x.device.type not in ("cuda", "musa"):
                return x
            return _zero_copy_alias(x)
        if isinstance(x, tuple):
            return tuple(make_weak_ref(i) for i in x)
        if isinstance(x, list):
            return [make_weak_ref(i) for i in x]
        if isinstance(x, dict):
            return {k: make_weak_ref(v) for k, v in x.items()}
        if x is None or isinstance(x, (int, float, bool)):
            return x
        raise TypeError(
            f"Invalid type {type(x).__name__} to make weak ref. Valid types are: "
            "torch.Tensor, tuple, list, dict, int, float, bool, and None."
        )

    make_weak_ref.__megatron_musa_patch_port_of__ = (
        "NVIDIA/TransformerEngine transformer_engine/pytorch/utils.py::make_weak_ref"
    )
    return make_weak_ref


def _weak_ref_self_check(make_weak_ref: Any, device: Any = None) -> str | None:
    """Exercise every contract Megatron puts on weak-referenced buffers.

    Returns a failure reason, or ``None`` when the port behaves on this stack.
    ``device`` defaults to the CUDA-spelled current device (the compat layer's
    spelling at hook time); hardware workers pass a MUSA device explicitly.
    Uses only ``zeros``/``ones`` factories, so no training RNG is consumed.
    """
    import torch

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    try:
        original = torch.zeros(4, device=device)
        ref = make_weak_ref(original)
        if not torch.is_tensor(ref):
            return "make_weak_ref did not return a tensor"
        if ref.data_ptr() != original.data_ptr():
            return "the weak ref is not a zero-copy view of the input"
        if tuple(ref.shape) != tuple(original.shape) or ref.dtype != original.dtype:
            return "the weak ref changed shape or dtype"
        if ref.grad_fn is not None:
            return "the weak ref carries autograd history"
        # Megatron attaches attributes and flips requires_grad on the ref.
        ref.can_skip_replay_copy = True
        ref.requires_grad = True
        with torch.no_grad():
            ref.copy_(torch.ones_like(original))
            if not bool(torch.all(original == 1)):
                return "writes through the weak ref did not reach the input storage"
            if not torch.equal(ref.clone(), original):
                return "cloning through the weak ref diverged"
        # Replay backward clones wgrads held as weak refs.
        nested = make_weak_ref({"t": [original], "none": None, "flag": True, "n": 3})
        if not (
            torch.is_tensor(nested["t"][0])
            and nested["none"] is None
            and nested["flag"] is True
            and nested["n"] == 3
        ):
            return "container recursion is broken"
        cpu = torch.zeros(2)
        if make_weak_ref(cpu) is not cpu:
            return "CPU tensors must pass through unchanged"
    except Exception as exc:  # noqa: BLE001 - the reason is reported verbatim
        return f"{type(exc).__name__}: {exc}"
    return None


def _install_make_weak_ref() -> bool:
    """Provide MT-TE's missing ``make_weak_ref`` when the graph stack is real."""
    global _owned
    if _owned is not None:
        return False
    import sys

    if not _musa_fork_installed():
        return False
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    missing = _missing_graph_prerequisites()
    if missing:
        logger.info("megatron.te.make-weak-ref.graph-compat declined: %s", "; ".join(missing))
        return False
    import importlib

    try:
        te_utils = importlib.import_module("transformer_engine.pytorch.utils")
    except Exception as exc:  # noqa: BLE001 - let Megatron surface its own error
        logger.info(
            "megatron.te.make-weak-ref.graph-compat declined: "
            "TransformerEngine is not importable (%s)",
            exc,
        )
        return False
    if hasattr(te_utils, "make_weak_ref"):
        return False  # The vendor ships it now; the removal condition is met.
    make_weak_ref = _build_make_weak_ref()
    failure = _weak_ref_self_check(make_weak_ref)
    if failure is not None:
        logger.info(
            "megatron.te.make-weak-ref.graph-compat declined: "
            "the make_weak_ref self-check failed (%s)",
            failure,
        )
        return False
    te_utils.make_weak_ref = make_weak_ref
    _owned = (te_utils, make_weak_ref)
    return True


def _uninstall_make_weak_ref() -> None:
    global _owned
    if _owned is None:
        return
    module, replacement = _owned
    if getattr(module, "make_weak_ref", None) is replacement:
        del module.make_weak_ref
    _owned = None


PATCHES = (
    HookPatch(
        id="megatron.te.make-weak-ref.graph-compat",
        trigger="megatron.core.parallel_state",
        run=_install_make_weak_ref,
        undo=_uninstall_make_weak_ref,
        rationale=(
            "Megatron core_v0.16.1 opens its whole TE graph surface inside one "
            "try block in megatron/core/transformer/cuda_graphs.py, including "
            "transformer_engine.pytorch.utils.make_weak_ref (newer NVIDIA TE). "
            "The installed MT-TE 2.0 fork lacks that one symbol, so the bare "
            "except leaves HAVE_TE_GRAPHS=False: CudaGraphManager fails its "
            "'RNG tracker does not support cudagraphs!' assertion and "
            "TECudaGraphHelper fails 'CUDA Graphs are not supported without "
            "TE.' -- the r2 sweep's 23 failures plus 3 errors in "
            "transformer/test_cuda_graphs.py are this import gate, before any "
            "real capture is attempted. Verified live: with the port installed, "
            "the module imports with HAVE_TE_GRAPHS True on torch_musa, whose "
            "MUSAGraph natively implements register_generator_state."
        ),
        strategy=(
            "Install a port of NVIDIA's make_weak_ref into "
            "transformer_engine.pytorch.utils only when the full prerequisite "
            "set is really present: a graph class that can register generator "
            "states, the graph/pool/stream capture surface, graph-safe "
            "Generator state APIs, TE's graph_safe_rng_available(), and the TE "
            "graph imports Megatron consumes -- any gap declines with the "
            "reason instead of faking capability. The port recurses containers "
            "and, for CUDA/MUSA tensors, returns a zero-copy alias of the same "
            "memory: NVIDIA's non-owning __cuda_array_interface__ path cannot "
            "work on torch builds without CUDA bindings ('Cannot initialize "
            "CUDA without ATen_cuda library', verified on torch_musa 2.11), so "
            "the alias shares storage -- same data_ptr/shape/dtype/strides, no "
            "autograd history, attribute attachment and clone intact; the "
            "deviation is buffer lifetime (owning), i.e. the higher-memory "
            "behavior Megatron already documents for runs without TE's weak "
            "references. A live-device self-check exercises the alias, "
            "attribute, clone and container contracts before install; undo "
            "deletes only the attribute this hook owns."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/core/transformer/cuda_graphs.py:42-58 "
            "(try-import incl. make_weak_ref); NVIDIA/TransformerEngine "
            "transformer_engine/pytorch/utils.py make_weak_ref/_WeakRefTensor"
        ),
        remove_when=(
            "The installed TransformerEngine provides "
            "transformer_engine.pytorch.utils.make_weak_ref (or Megatron stops "
            "importing it): disable this id, confirm "
            "megatron.core.transformer.cuda_graphs imports with HAVE_TE_GRAPHS "
            "true without the shim, re-run "
            "tests/unit_tests/transformer/test_cuda_graphs.py on MUSA, then "
            "delete."
        ),
    ),
)
