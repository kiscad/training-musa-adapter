"""Error types raised by :mod:`training_musa_adaptor`.

Everything derives from :class:`TrainingMusaAdaptorError` so callers can
catch one base class.  Errors name the exact patch, target and detected
condition: this package patches fast-moving third-party libraries, and a
bare ``AttributeError`` deep inside a training step is not an acceptable
failure mode.
"""

from __future__ import annotations

__all__ = [
    "TrainingMusaAdaptorError",
    "PatchTargetMissing",
    "PatchConflict",
    "ConfigError",
    "EngineOverlapError",
    "WatcherInstallError",
    "MusaUnavailable",
]


class TrainingMusaAdaptorError(RuntimeError):
    """Base class for every error raised by this package."""


class PatchTargetMissing(TrainingMusaAdaptorError):
    """A patch target no longer exists in the installed upstream library."""

    def __init__(self, patch_id: str, target: str, detail: str = ""):
        self.patch_id = patch_id
        self.target = target
        message = (
            f"patch {patch_id!r} cannot be applied: {target!r} was not found. "
            "The upstream package probably renamed or removed this symbol "
            "(symbol drift in a matched version is a compatibility failure)."
        )
        if detail:
            message += f" [{detail}]"
        super().__init__(message)


class PatchConflict(TrainingMusaAdaptorError):
    """Something else already claimed the attribute we wanted to patch."""


class ConfigError(TrainingMusaAdaptorError):
    """Invalid configuration: unknown field/ID/implementation/enum/type.

    Config errors are hard failures at the first relevant import boundary;
    nothing silently falls back to defaults (design doc §7.1).
    """


class EngineOverlapError(TrainingMusaAdaptorError):
    """A known legacy patch engine is active in this process.

    megatron-musa-patch and the first-round musa-adapter engine must not be
    active simultaneously with training-musa-adaptor (design doc §9.3):
    two engines writing the same targets is not a supported scenario.
    """


class WatcherInstallError(TrainingMusaAdaptorError):
    """The import watcher itself could not be installed.

    Without the watcher there is no reliable interception point; diagnostics
    must surface this loudly instead of pretending coverage.
    """


class MusaUnavailable(TrainingMusaAdaptorError):
    """torch_musa is missing, or no MUSA device is visible."""
