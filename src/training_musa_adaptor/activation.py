"""Activation channels (design doc §5.1/§7.3).

Exactly two ways this package comes alive, both ending in the same
idempotent :meth:`~training_musa_adaptor._engine.Engine.install`:

1. **Automatic** -- the ``torch.backends`` entry point declared in
   ``pyproject.toml``.  PyTorch imports every entry point in that group at
   the very end of ``import torch``.  Kill switches:
   ``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` (torch's own) and
   ``TRAINING_MUSA_ADAPTOR_AUTOLOAD=0`` (ours).
2. **Explicit** -- :func:`training_musa_adaptor.install` (optionally with a
   config path), used by diagnostics, custom integration and tests.

``import training_musa_adaptor`` itself never installs anything.
"""

from __future__ import annotations

import logging

from . import _config as _config_mod
from ._engine import Engine

__all__ = [
    "install",
    "apply",
    "uninstall",
    "is_applied",
    "report",
    "torch_backend_autoload",
    "bootstrap_errors",
    "ENGINE",
]

logger = logging.getLogger("training_musa_adaptor")

ENGINE = Engine()

#: Initialization failures recorded by the automatic channel (never raised
#: there); surfaced by the next relevant boundary and by diagnostics.
bootstrap_errors: list[dict[str, str]] = []

_registered = False


def _register() -> None:
    global _registered
    if _registered:
        return
    from .patches import PATCHES

    ENGINE.register(PATCHES)
    _registered = True


def install(config_path: str | None = None) -> None:
    """Register the patch set and start watching for target imports.

    Cheap and safe to call from inside ``import torch``: it imports neither
    ``torch`` nor any framework and only adds one finder to
    ``sys.meta_path``.  With ``eager=True`` (the explicit path) the
    configuration is parsed and frozen now; already-imported attr targets
    are patched (late, with the documented caveats) and hooks whose
    boundary was missed are reported phase_missed.
    """
    if not ENGINE.config.enabled():
        return
    _register()
    ENGINE.install(config_path=config_path, eager=True)


def apply(patch_ids=()) -> None:
    """Import and apply only the named patches (diagnostic entry)."""
    if not ENGINE.config.enabled():
        return
    _register()
    ENGINE.apply(patch_ids)


def uninstall() -> None:
    """Restore the original objects and stop watching imports."""
    ENGINE.uninstall()


def is_applied(patch_id: str | None = None) -> bool:
    return ENGINE.is_applied(patch_id)


def report() -> dict:
    """Read-only description of every registered patch and the config."""
    data = ENGINE.report()
    if bootstrap_errors:
        data["bootstrap_errors"] = list(bootstrap_errors)
    return data


def torch_backend_autoload() -> None:
    """Entry point for the ``torch.backends`` group -- must never raise.

    Runs inside ``import torch``; an exception here would break every
    process in the environment.  Only the two early switches are read and
    the lightweight watcher is installed; config parsing is deferred to
    the first relevant import boundary, where a saved initialization error
    is then raised (design doc §5.5).
    """
    if not ENGINE.config.autoload_enabled():
        logger.debug("TRAINING_MUSA_ADAPTOR_AUTOLOAD=0 -- automatic channel off")
        return
    try:
        if not ENGINE.config.enabled():
            return
        _register()
        ENGINE.install(eager=False)
    except Exception as exc:  # noqa: BLE001 - never break `import torch`
        bootstrap_errors.append({"type": type(exc).__name__, "message": str(exc)})
        import sys

        if ENGINE._watcher in sys.meta_path:
            # The watcher survived: the saved error is raised at the first
            # relevant boundary (design doc §5.5).
            ENGINE._saved_error = exc
        logger.exception(
            "training-musa-adaptor failed to initialize; the error will be "
            "raised at the first relevant import boundary."
        )
