"""Transformer Engine call-shape adapters for the MUSA port.

Migrated from megatron-musa-patch ``patches/_transformer_engine.py`` (rev
a1090de). The two ``trigger="megatron"`` hooks now use the concrete
``megatron.core.parallel_state`` boundary (megatron is a namespace package
and not an executable boundary; parallel_state is reachable in both import
orders -- see patches/platform.py's boundary analysis).

The MUSA Transformer Engine fork reports NVIDIA TE release numbering without
following NVIDIA's release API timeline, so Megatron's version thresholds do
not describe the installed API.  This module no longer overrides the version
predicate itself: ``megatron.core.utils.is_te_min_version`` keeps its real
comparison so that version-specific branches report the fork honestly.
Call-shape gaps are instead adapted at their own boundary by inspecting the
installed function (see ``_cpu_offload_context_by_signature``), or reported as
unsupported (QK-clip max-logit, ``quantized_model_init``).
"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from threading import RLock
from typing import Any

from .. import _compat
from .._engine import AttrPatch, HookPatch

__all__ = ["PATCHES"]

logger = logging.getLogger("training_musa_adaptor")

#: Megatron's Transformer Engine extension module: wraps TE and owns the
#: version-gated call shapes this module adapts.
_TE_EXTENSION = "megatron.core.extensions.transformer_engine"

#: sys.modules entries this package installs for the MT-TE fork's legacy
#: ``musa_patch`` dependency, and that its undo may remove.
_mem_monitor_owned: dict[str, Any] = {}
_quantized_init_owned: tuple[Any, Any] | None = None


def _install_quantized_model_init() -> bool:
    """Expose only the verified delayed-FP8 subset of the newer TE spelling.

    Runs after Megatron activation, never during patch registration. The
    attribute is owned by this hook, not by a second import/patch system.
    """
    global _quantized_init_owned
    import importlib
    import inspect
    import sys

    if _quantized_init_owned is not None or not _te_fork_needs_mem_monitor():
        return False
    # Device adaptation is a runtime prerequisite, not a reason to initialize
    # the vendor stack inside a CPU-only or synthetic Megatron import.
    torch = sys.modules.get("torch")
    musa = getattr(torch, "musa", None)
    if musa is None or not musa.is_available() or not torch.cuda.is_available():
        return False
    te = importlib.import_module("transformer_engine.pytorch")
    if hasattr(te, "quantized_model_init"):
        return False
    original = getattr(te, "fp8_model_init", None)
    if original is None:
        return False
    signature = inspect.signature(original)
    if not {"enabled", "recipe", "preserve_high_precision_init_val"} <= signature.parameters.keys():
        return False
    from transformer_engine.common.recipe import DelayedScaling

    @functools.wraps(original)
    def quantized_model_init(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        if arguments.arguments["enabled"] and not isinstance(
            arguments.arguments["recipe"], DelayedScaling
        ):
            raise NotImplementedError(
                "MUSA quantized_model_init compatibility requires an explicit "
                "DelayedScaling recipe; other quantization recipes need native TE support"
            )
        return original(*arguments.args, **arguments.kwargs)

    te.quantized_model_init = quantized_model_init
    _quantized_init_owned = (te, quantized_model_init)
    return True


def _uninstall_quantized_model_init() -> None:
    global _quantized_init_owned
    if _quantized_init_owned is not None:
        module, replacement = _quantized_init_owned
        if getattr(module, "quantized_model_init", None) is replacement:
            del module.quantized_model_init
        _quantized_init_owned = None


# Early TE hooks run before Megatron, but only for the installed MUSA fork.
_jit_script_owned: tuple[Any, Any, Any] | None = None
_jit_compile_lock = RLock()
_FACTORY_NAMES = ("tensor", "zeros", "ones", "empty", "rand", "arange", "empty_like")


@contextmanager
def _script_factory_aliases(torch_module):
    """Teach TorchScript the ATen identity of known MT-TE factory wrappers.

    Never replace torch's eager bindings: another thread may allocate tensors
    while compilation is in progress. Only the compiler's builtin table is
    temporarily updated. This private PyTorch API is covered by the real
    TorchScript regression and must be rechecked when upgrading PyTorch.
    Device arguments in compiled graphs retain ATen semantics; this does not
    add CUDA-string translation inside TorchScript graphs.
    """
    import types

    from torch.jit import _builtins

    missing = object()
    with _jit_compile_lock:
        try:
            table = _builtins._get_builtin_table()
        except Exception as exc:  # noqa: BLE001 - private API, torch-version bound
            logger.info(
                "factory-shim passthrough: torch.jit._builtins._get_builtin_table "
                "unavailable in this torch build (%s: %s)",
                type(exc).__name__,
                exc,
            )
            yield
            return
        changed = []
        try:
            for name in _FACTORY_NAMES:
                wrapper = getattr(torch_module, name, None)
                if (
                    type(wrapper) is not types.FunctionType
                    or wrapper.__module__ != "transformer_engine.musa"
                ):
                    continue
                closure = dict(zip(wrapper.__code__.co_freevars, wrapper.__closure__ or ()))
                cell = closure.get(f"original_{name}")
                if cell is None:
                    continue
                try:
                    original = cell.cell_contents
                except ValueError:  # Empty closure cell: not this vendor contract.
                    continue
                if original is not getattr(torch_module._C._VariableFunctions, name, None):
                    continue
                key = id(wrapper)
                previous = table.get(key, missing)
                op = f"aten::{name}"
                if previous is not missing:
                    # Respect an existing compiler mapping, including another owner.
                    continue
                changed.append((key, wrapper, op))  # Keep function identities alive.
                table[key] = op
            yield
        finally:
            for key, _wrapper, op in reversed(changed):
                if table.get(key) is op:
                    del table[key]


def _install_jit_script_compat() -> bool:
    """Adapt scripting without temporarily disabling eager device translation."""
    global _jit_script_owned
    if _jit_script_owned is not None or not _te_fork_needs_mem_monitor():
        return False
    shimmed = _compat.module_source_contains(
        "transformer_engine.musa", "torch.arange = patched_arange"
    )
    if shimmed is False:
        # The installed TE no longer wraps factory functions: the alias guard
        # would never register anything, so skip owning torch.jit.script.
        logger.info(
            "factory-shim skipped: transformer_engine.musa has no factory "
            "wrappers (te=%s torch_musa=%s)",
            _compat.distribution_version("transformer-engine"),
            _compat.distribution_version("torch-musa"),
        )
        return False
    import inspect

    import torch

    original = torch.jit.script
    signature = inspect.signature(original)

    @functools.wraps(original)
    def script(*args: Any, **kwargs: Any):
        arguments = signature.bind(*args, **kwargs)
        # Class scripting resolves names from the caller's frame. Account for
        # this wrapper just as torch.jit.script accounts for its own frame.
        if "_frames_up" in signature.parameters:
            arguments.arguments["_frames_up"] = arguments.arguments.get("_frames_up", 0) + 1
        with _script_factory_aliases(torch):
            return original(*arguments.args, **arguments.kwargs)

    torch.jit.script = script
    _jit_script_owned = (torch.jit, original, script)
    return True


def _uninstall_jit_script_compat() -> None:
    global _jit_script_owned
    if _jit_script_owned is not None:
        owner, original, replacement = _jit_script_owned
        if owner.script is replacement:
            owner.script = original
        _jit_script_owned = None


#: sys.modules entry this hook seeds for the vendor's unsafe utils module.
_utils_module_owned: tuple[str, Any] | None = None


def _install_safe_te_utils_module() -> bool:
    """Seed a safe replica of ``transformer_engine.musa.pytorch.utils``.

    The vendor module iterates ``sys.modules`` and getattr's every 'utils'
    module at import time. With transformers 5.x lazy modules that getattr
    triggers imports and crashes the whole TE import chain ("dictionary
    changed size during iteration"). The functions TE actually consumes
    (wrap_attr/replace_attr/add_attr) are trivial, so a replica without the
    fragile loop is seeded before the vendor module executes. Refuses to
    shadow an existing entry and removes only its own seeding on undo.
    """
    import sys
    import types

    global _utils_module_owned
    name = "transformer_engine.musa.pytorch.utils"
    if name in sys.modules or _utils_module_owned is not None or not _te_fork_needs_mem_monitor():
        return False
    unsafe = _compat.module_source_contains(
        "transformer_engine.musa.pytorch.utils",
        "for k in sys.modules:",
        "getattr(sys.modules[k], target, None)",
    )
    if unsafe is False:
        # The installed TE fixed the fragile loop: seeding would shadow the
        # vendor's repaired module with this replica, so decline instead.
        logger.info(
            "safe-seed skipped: transformer_engine utils module no longer "
            "iterates sys.modules (te=%s)",
            _compat.distribution_version("transformer-engine"),
        )
        return False

    module = types.ModuleType(name)
    module.__package__ = name.rpartition(".")[0]
    from importlib.machinery import ModuleSpec

    module.__spec__ = ModuleSpec(name, loader=None)

    def wrap_name(src_name: str) -> str:
        return f"_orig_{src_name}"

    def add_attr(module_: Any, attr: str, target: Any) -> None:
        setattr(module_, attr, target)

    def wrap_attr(module_: Any, attr: str, wrapper: Any) -> None:
        target = getattr(module_, attr)
        setattr(module_, wrap_name(attr), target)
        setattr(module_, attr, wrapper)

    def replace_attr(module_: Any, attr: str, target: Any) -> None:
        wrap_attr(module_, attr, target)

    def musa_assert_dim_for_fp8_exec(*tensors: Any) -> None:
        return None

    module.wrap_name = wrap_name
    module.add_attr = add_attr
    module.wrap_attr = wrap_attr
    module.replace_attr = replace_attr
    module.musa_assert_dim_for_fp8_exec = musa_assert_dim_for_fp8_exec
    sys.modules[name] = module
    _utils_module_owned = (name, module)
    return True


def _uninstall_safe_te_utils_module() -> None:
    import sys

    global _utils_module_owned
    if _utils_module_owned is not None:
        name, module = _utils_module_owned
        if sys.modules.get(name) is module:
            del sys.modules[name]
        parent_name, _, child = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and vars(parent).get(child) is module:
            delattr(parent, child)
        _utils_module_owned = None


def _te_extension():
    """The extension module being patched, without importing it.

    It is always in ``sys.modules`` when these patches run; importing it here
    would drag Transformer Engine (and, on MUSA, its shared libraries) into
    processes and unit tests that never needed it.
    """
    import sys

    return sys.modules.get(_TE_EXTENSION)


def _cpu_offload_context_by_signature(original: Any) -> Any:
    """Pick TE's CPU-offload call by arity, not by version threshold.

    Megatron chooses between Transformer Engine's three CPU-offload signatures
    with ``is_te_min_version`` thresholds, so a fork whose reported version and
    real API disagree is called with the wrong argument count:
    ``TransformerBlock.__init__`` -- which calls this unconditionally -- raises
    ``TypeError: ... takes from 0 to 5 positional arguments but 6 were given``
    before the model is built.  Dispatch on the installed function's real
    signature instead; leave upstream alone when it is already right.
    """
    import inspect

    module = _te_extension()
    target = getattr(module, "_get_cpu_offload_context", None) if module else None
    if target is None or original is None:
        return None
    try:
        parameters = list(inspect.signature(target).parameters.values())
    except (TypeError, ValueError):
        return None
    if any(p.kind is p.VAR_POSITIONAL for p in parameters):
        return None
    accepted = sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in parameters)
    if accepted != 5:  # >= 6: upstream's choice is right; fewer: not this fork
        return None

    @functools.wraps(original)
    def get_cpu_offload_context(
        enabled,
        num_layers,
        model_layers,
        activation_offloading,
        weight_offloading,
        double_buffering,
    ):
        """Get CPU offload context and sync function (five-argument TE)."""
        return target(enabled, num_layers, model_layers, activation_offloading, weight_offloading)

    return get_cpu_offload_context


def _te_fork_needs_mem_monitor() -> bool:
    """Is the installed TransformerEngine the MUSA fork that imports musa_patch?"""
    import importlib.metadata as md
    from pathlib import Path

    try:
        distribution = md.distribution("transformer_engine")
        musa_dir = Path(str(distribution.locate_file("transformer_engine/musa")))
    except Exception:  # noqa: BLE001 - distribution metadata is best effort
        return False
    return musa_dir.exists()


def _install_mem_monitor_shim() -> bool:
    """Bridge the MT-TE fork's legacy ``musa_patch.mem_utils`` import.

    ``transformer_engine/pytorch/module/grouped_linear.py`` imports
    ``musa_patch.mem_utils.MemMonitor`` unconditionally and updates
    ``MemMonitor.max_token_num``; the legacy ``musa_patch`` distribution is not
    installed. Provide the minimal accounting type the fork mutates, owned by
    this hook so undo removes exactly what it added.
    """
    import sys
    import types

    if "musa_patch" in sys.modules or "musa_patch.mem_utils" in sys.modules:
        # A real musa_patch (or another owner) is loaded: never shadow it.
        return False
    if _compat.find_spec_without_watchers("musa_patch") is not None:
        return False  # Respect installed packages without executing their imports.
    if not _te_fork_needs_mem_monitor():
        return False

    package = types.ModuleType("musa_patch")
    package.__path__ = []  # mark as a package so submodule imports resolve
    package.__doc__ = "Compatibility shim owned by megatron-musa-patch."

    mem_utils = types.ModuleType("musa_patch.mem_utils")
    mem_utils.__doc__ = "Shim for the MT-TE fork's grouped-linear memory accounting."

    class MemMonitor:
        """Tracks the largest token count seen, as the fork's import expects."""

        max_token_num = 0

    mem_utils.MemMonitor = MemMonitor
    package.mem_utils = mem_utils

    sys.modules["musa_patch"] = package
    sys.modules["musa_patch.mem_utils"] = mem_utils
    _mem_monitor_owned["musa_patch"] = package
    _mem_monitor_owned["musa_patch.mem_utils"] = mem_utils
    return True


def _uninstall_mem_monitor_shim() -> None:
    import sys

    for name in list(_mem_monitor_owned):
        module = _mem_monitor_owned.pop(name)
        if sys.modules.get(name) is module:
            del sys.modules[name]


PATCHES = (
    HookPatch(
        id="megatron.te.quantized-model-init.delayed-compat",
        trigger="megatron.core.parallel_state",
        run=_install_quantized_model_init,
        undo=_uninstall_quantized_model_init,
        rationale=(
            "MT-TE 2.0 exposes fp8_model_init but FSDP delayed-FP8 callers use "
            "quantized_model_init with preserve_high_precision_init_val."
        ),
        strategy=(
            "After Megatron activation, own the missing TE attribute and delegate "
            "explicit DelayedScaling contexts to the existing native implementation. "
            "Reject other enabled recipes, preserve context nesting and exceptions, "
            "and never overwrite a native or third-party implementation."
        ),
        upstream="TransformerEngine pytorch/fp8.py:fp8_model_init",
        remove_when=(
            "Remove once native quantized_model_init supports delayed FP8 with "
            "high-precision initialization and both original FSDP cases pass."
        ),
    ),
    AttrPatch(
        id="megatron.te.cpu-offload-context.signature-dispatch",
        target=f"{_TE_EXTENSION}:get_cpu_offload_context",
        replace=_cpu_offload_context_by_signature,
        rationale=(
            "TransformerBlock calls get_cpu_offload_context unconditionally, and "
            "Megatron picks TE's six-argument (TE >= 2.5) call from a version "
            "threshold. The MUSA TE fork reports release numbering without "
            "following NVIDIA's API timeline, so its reported version can select "
            "the six-argument call while "
            "transformer_engine.pytorch.cpu_offload.get_cpu_offload_context still "
            "takes five arguments (no double_buffering); every run that builds a "
            "TransformerBlock then dies with 'takes from 0 to 5 positional "
            "arguments but 6 were given' before the model exists -- on a path that "
            "never touches CPU offloading. Observed on the MT fork of TE 2.0.0 "
            "with Megatron core_v0.16.1."
        ),
        strategy=(
            "Keep Megatron's public six-argument wrapper and call the installed TE "
            "function with the five arguments it accepts, chosen by inspecting that "
            "function's own signature instead of a version number. Declines when the "
            "installed function takes six arguments (upstream's choice is then correct) "
            "or when its arity is anything else, so the patch never guesses."
        ),
        upstream=(
            "NVIDIA/Megatron-LM megatron/core/extensions/transformer_engine.py:"
            "get_cpu_offload_context"
        ),
        remove_when=(
            "Remove once the MUSA Transformer Engine exposes the double_buffering "
            "argument, after a pretrain run with CPU offloading enabled and disabled "
            "has been re-validated on MUSA."
        ),
    ),
    HookPatch(
        id="megatron.te.grouped-linear.mem-monitor-compat",
        trigger="megatron.core.parallel_state",
        run=_install_mem_monitor_shim,
        undo=_uninstall_mem_monitor_shim,
        rationale=(
            "The MUSA TransformerEngine fork's grouped_linear.py imports "
            "musa_patch.mem_utils.MemMonitor unconditionally inside "
            "TEGroupedLinear._forward and updates MemMonitor.max_token_num, but "
            "the legacy musa_patch distribution is not installed, so every MoE "
            "grouped-GEMM forward raises ModuleNotFoundError before any math. "
            "Observed on MT-TE 2.0.0 with Megatron core_v0.16.1 in "
            "a2a_overlap and dist_checkpointing grouped-expert cases."
        ),
        strategy=(
            "When Megatron is first imported, the import name is free and the "
            "installed TransformerEngine is the MUSA fork, own two sys.modules "
            "entries (musa_patch and musa_patch.mem_utils) carrying the minimal "
            "MemMonitor type with a max_token_num counter, so the fork's "
            "accounting update keeps its behavior instead of becoming a no-op "
            "mock. A pre-existing musa_patch module is never shadowed; a failing "
            "install removes its partial entries; undo deletes only modules this "
            "hook still owns. This is a bridge for one TE import, not a "
            "re-creation of the legacy package."
        ),
        upstream="transformer_engine/pytorch/module/grouped_linear.py:musa_patch import",
        remove_when=(
            "Remove when the MUSA TransformerEngine drops the external "
            "musa_patch import: disable this patch id and re-run the MoE/A2A "
            "grouped-linear forward cases; delete only if they pass without it."
        ),
    ),
    HookPatch(
        id="megatron.te.factory-shim.torchscript-compat",
        trigger="transformer_engine",
        run=_install_jit_script_compat,
        undo=_uninstall_jit_script_compat,
        rationale=(
            "MT-TE's patch_after_import_torch rebinds torch.tensor/zeros/ones/"
            "empty/rand/arange/empty_like to untyped ``*args, **kwargs`` "
            "wrappers that translate device arguments. TorchScript then "
            "rejects any eager ``torch.jit.script`` of a function calling one "
            "of those factories with torch.jit.frontend.NotSupportedError; "
            "ms-swift hits this on ``import swift.megatron`` because "
            "swift.model imports sequence_parallel/zigzag_ring_attn, whose "
            "module-level @torch.jit.script compiles torch.arange calls "
            "before Megatron is ever imported. The wrappers themselves are "
            "load-bearing: swift, mcore-bridge and Megatron-Core pass eager "
            "``device='cuda'`` strings to factory calls on the training path."
        ),
        strategy=(
            "Only for the installed MUSA TE fork, own torch.jit.script at the "
            "TE import boundary. During scripting, register the seven known "
            "TE wrappers as their ATen builtins in TorchScript's compiler "
            "table, serialize nested/concurrent registration and undo owned "
            "entries even on compile failure. Eager torch factories are never "
            "rebound, including during compilation. Preserve caller-frame "
            "resolution and restore script only by exact identity on undo. "
            "Scripted devices retain ATen semantics: use actual device "
            "objects, not CUDA string literals expecting eager translation. "
            "Declines when the installed TE no longer wraps factory "
            "functions (source probe)."
        ),
        upstream="transformer_engine/musa/__init__.py:patch_after_import_torch",
        remove_when=(
            "Remove when MT-TE stops rebinding torch factory functions with "
            "untyped wrappers or ships scriptable typed wrappers: disable this "
            "patch id and re-run the standard swift.megatron import probe and "
            "the transformer-engine-bug-tests Bug 3 probes; delete only if "
            "both pass without it."
        ),
    ),
    HookPatch(
        id="megatron.te.utils-module.safe-seed",
        trigger="transformer_engine",
        run=_install_safe_te_utils_module,
        undo=_uninstall_safe_te_utils_module,
        rationale=(
            "MT-TE's transformer_engine/musa/pytorch/utils.py iterates "
            "sys.modules at import time and getattr's every module whose name "
            "contains 'utils'. With transformers 5.x lazy modules that "
            "getattr triggers submodule imports, mutating sys.modules during "
            "iteration, and the whole TE import chain dies with 'dictionary "
            "changed size during iteration' -- which breaks "
            "``import swift.megatron`` after the transformers upgrade."
        ),
        strategy=(
            "Only for the installed MUSA fork, before the vendor module executes, "
            "seed sys.modules with a "
            "replica exposing the functions TE consumes (wrap_attr, "
            "replace_attr, add_attr, wrap_name, "
            "musa_assert_dim_for_fp8_exec) without the fragile import-time "
            "loop, so the vendor module body never runs. Same seeding "
            "pattern as the musa_patch.mem_utils shim. Undo deletes the "
            "entry only if this hook still owns it; refuses to shadow an "
            "existing module. Declines when the vendor module no longer "
            "contains the fragile loop (source probe)."
        ),
        upstream="transformer_engine/musa/pytorch/utils.py",
        remove_when=(
            "Remove when the MUSA TE stops iterating sys.modules at import "
            "time: disable this patch id and re-run the standard "
            "swift.megatron import probe with transformers 5.x; delete only "
            "if it passes without it."
        ),
    ),
)
