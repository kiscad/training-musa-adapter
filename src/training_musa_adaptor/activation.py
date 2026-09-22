"""Activation channels.

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
    from .patches import PATCH_SUITES, PATCHES

    ENGINE.register(PATCHES, patch_suites=PATCH_SUITES)
    _registered = True


def install(config_path: str | None = None) -> None:
    """Register the patch set and start watching for target imports.

    This explicit path parses and freezes configuration immediately.
    Already-imported attr targets
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
    import sys

    data = ENGINE.report()
    ops = sys.modules.get("training_musa_adaptor.ops.attention")
    if ops is not None:
        data["last_attention_dispatch"] = ops.dispatch_report()
    if bootstrap_errors:
        data["bootstrap_errors"] = list(bootstrap_errors)
    return data


def torch_backend_autoload() -> None:
    """Entry point for the ``torch.backends`` group -- must never raise.

    Runs inside ``import torch``; an exception here would break every
    process in the environment.  Only the two early switches are read and
    the lightweight watcher is installed; config parsing is deferred to
    the first relevant import boundary, where a saved initialization error
    is then raised.
    """
    try:
        if not ENGINE.config.autoload_enabled():
            logger.debug("TRAINING_MUSA_ADAPTOR_AUTOLOAD=0 -- automatic channel off")
            return
        _register()
        ENGINE.install(eager=False)
    except Exception as exc:  # noqa: BLE001 - never break `import torch`
        bootstrap_errors.append({"type": type(exc).__name__, "message": str(exc)})
        import sys

        ENGINE._saved_error = exc
        if ENGINE._watcher in sys.meta_path:
            detail = "the error will be raised at the first relevant import boundary"
        else:
            detail = "no watcher is available; automatic adaptation is unavailable"
        logger.exception("training-musa-adaptor failed to initialize; %s", detail)
