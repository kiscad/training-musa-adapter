"""training-musa-adaptor -- adapt upstream training frameworks to MUSA GPUs.

Quick start: install the package and do nothing else::

    pip install --no-deps .
    torchrun --nproc_per_node=8 pretrain_gpt.py ...

The ``torch.backends`` entry point makes PyTorch call
:func:`training_musa_adaptor.activation.torch_backend_autoload` at the end
of ``import torch``.  ``import training_musa_adaptor`` itself is
**lightweight and never patches anything** (design doc §7.3); the explicit
API is for diagnostics, custom integration and tests::

    import training_musa_adaptor as tma
    tma.install()
    records = tma.report()
    tma.apply(patch_ids=("transformers.qwen3-vl.text-rms-norm.fused-torch",))
    tma.uninstall()

Switches::

    TRAINING_MUSA_ADAPTOR_ENABLED=0       disable everything (hard exit)
    TRAINING_MUSA_ADAPTOR_AUTOLOAD=0      disable only the automatic channel
    TRAINING_MUSA_ADAPTOR_ONLY=a,b        patch ID whitelist (wins over DISABLE)
    TRAINING_MUSA_ADAPTOR_DISABLE=a,b     patch ID denylist
    TRAINING_MUSA_ADAPTOR_DEBUG=1         log every applied patch at INFO
"""

from __future__ import annotations

from . import _config as _config  # noqa: F401  (env-var documentation home)
from ._engine import AppliedPatch, AttrPatch, HookPatch
from ._errors import (
    ConfigError,
    EngineOverlapError,
    MusaUnavailable,
    PatchConflict,
    PatchTargetMissing,
    TrainingMusaAdaptorError,
    WatcherInstallError,
)
from .activation import (
    apply,
    bootstrap_errors,
    install,
    is_applied,
    report,
    torch_backend_autoload,
    uninstall,
)


def _distribution_version() -> str:
    """Read our own version from the installed metadata (pyproject is the
    single source of truth); falls back to a marker for bare checkouts."""
    import importlib.metadata as md

    try:
        return md.version("training-musa-adaptor")
    except md.PackageNotFoundError:  # pragma: no cover - uninstalled checkout
        return "0.0.0+unknown"


__version__ = _distribution_version()

__all__ = [
    "__version__",
    # activation
    "install",
    "apply",
    "uninstall",
    "is_applied",
    "report",
    "torch_backend_autoload",
    "bootstrap_errors",
    # patch author interface
    "AttrPatch",
    "HookPatch",
    "AppliedPatch",
    # errors
    "TrainingMusaAdaptorError",
    "PatchTargetMissing",
    "PatchConflict",
    "ConfigError",
    "EngineOverlapError",
    "WatcherInstallError",
    "MusaUnavailable",
]
