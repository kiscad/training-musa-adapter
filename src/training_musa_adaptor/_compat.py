"""Compatibility helpers: target resolution, version gates, source checks.

Standard-library only at import time: this module runs inside ``import
torch`` via the ``torch.backends`` entry point.  ``packaging`` is imported
lazily at the first real patch boundary (design doc §7.2), never here.
"""

from __future__ import annotations

import importlib.metadata as md
import importlib.util
import logging
import re
import sys
from typing import Any

from ._errors import PatchTargetMissing, TrainingMusaAdaptorError
from ._imports import find_spec_without_watchers  # noqa: F401  (re-export)

__all__ = [
    "logger",
    "split_target",
    "require_attr",
    "distribution_version",
    "module_source_contains",
    "find_spec_without_watchers",
    "normalize_distribution_name",
    "parse_version_gate",
    "check_version_gate",
    "validate_version_gates",
]

logger = logging.getLogger("training_musa_adaptor")

#: Distribution names whose ``import`` name differs from the pip name and is
#: recorded in reports (design doc §7.2).
_IMPORT_NAMES: dict[str, str] = {
    "torch-musa": "torch_musa",
    "transformer-engine": "transformer_engine",
    "flash-attn": "flash_attn",
    "flash-linear-attention": "flash_linear_attention",
    "megatron-core": "megatron.core",
    "megatron-lm": "megatron",
    "ms-swift": "swift",
    "mcore-bridge": "mcore_bridge",
    "torch-kernels": "torch_kernels",
}


def normalize_distribution_name(name: str) -> str:
    """PEP 503 normalized distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def distribution_version(distribution: str) -> str | None:
    """Installed version of a distribution, or None when absent.

    Uses import metadata on purpose: importing the package to ask its
    version would drag torch (and possibly the device) into processes that
    only wanted to inspect the environment.
    """
    try:
        return md.version(normalize_distribution_name(distribution))
    except md.PackageNotFoundError:
        return None


def split_target(target: str) -> tuple[str, str]:
    """Split ``"package.module:Class.method"`` into (module, attribute)."""
    module, _, attr = target.partition(":")
    if not module or not attr:
        raise ValueError(f"target {target!r} must look like 'package.module:attribute'")
    if not all(part.isidentifier() for part in module.split(".")):
        raise ValueError(f"target {target!r} has an invalid module path")
    if not all(part.isidentifier() for part in attr.split(".")):
        raise ValueError(f"target {target!r} has an invalid attribute path")
    return module, attr


def require_attr(module: Any, attribute: str, *, patch_id: str, target: str) -> Any:
    """Resolve ``attribute`` (possibly dotted) on ``module`` or fail loudly."""
    owner: Any = module
    for part in attribute.split("."):
        try:
            owner = getattr(owner, part)
        except AttributeError as exc:
            raise PatchTargetMissing(patch_id, target, f"cannot resolve {attribute!r}") from exc
    return owner


def module_source_contains(module_name: str, *markers: str) -> bool | None:
    """Check markers in a module's source **without executing the module**.

    Several MUSA-fork defects this package works around are identified by
    exact code patterns whose presence depends on the vendor build, and the
    vendor version strings do not follow the upstream API timeline.  Reading
    source markers provide a conservative build fingerprint, not proof of
    kernel correctness. The probe never runs the module body.

    Dotted names are resolved from the top-level package's spec by joining
    the remaining parts as file paths -- resolving a submodule through the
    import system would import (and execute) its parents.

    Returns ``True`` when every marker appears in the source, ``False`` when
    the source is readable and any marker is missing, and ``None`` when the
    probe cannot decide (unsupported layout or unreadable source).
    A missing package or submodule returns False.
    Callers choose their own policy for the unknown case.
    """
    if not markers:
        raise ValueError("module_source_contains requires at least one marker")
    parts = module_name.split(".")
    try:
        spec = find_spec_without_watchers(parts[0])
    except Exception:  # noqa: BLE001 - a broken lookup is an undecided probe
        return None
    if spec is None:
        return False  # the top-level package is genuinely absent
    origin = getattr(spec, "origin", None)
    if not origin or not origin.endswith(".py"):
        return None  # namespace/zip layout: undecidable without executing
    from pathlib import Path

    if len(parts) == 1:
        target = Path(origin)
    elif not getattr(spec, "submodule_search_locations", None):
        return False
    else:
        target = Path(origin).parent.joinpath(*parts[1:])
    if target.is_dir():
        target = target / "__init__.py"
    elif not target.exists():
        target = (
            target.with_name(target.name + ".py") if not target.name.endswith(".py") else target
        )
    if not target.exists():
        return False  # submodule file genuinely absent
    try:
        with open(target, encoding="utf-8", errors="replace") as handle:
            source = handle.read()
    except OSError:
        return None
    return all(marker in source for marker in markers)


# ---------------------------------------------------------------------------
# Version gates (design doc §7.2)
# ---------------------------------------------------------------------------

_GATE_RE = re.compile(r"^\s*(?P<name>[A-Za-z0-9_.\-]+?)\s*(?P<spec><=|>=|==|!=|<|>|~=).*$")


def parse_version_gate(gate: str) -> tuple[str, str]:
    """Split ``"transformer_engine >=2.0,<2.1"`` into (distribution, specifier)."""
    match = _GATE_RE.match(gate)
    if not match:
        raise ValueError(
            f"version gate {gate!r} must look like 'distribution >=1.0,<2.0'"
        )
    return (
        normalize_distribution_name(match.group("name")),
        gate[match.start("spec"):].strip(),
    )


def validate_version_gates(gates: tuple[str, ...]) -> None:
    for gate in gates:
        name, specifier = parse_version_gate(gate)
        _specifier_set(name, specifier)  # eager syntax validation only


def _specifier_set(distribution: str, specifier: str):
    """Build a SpecifierSet with prereleases explicitly included.

    Design doc §7.2: prerelease handling is explicit (``prereleases=True``)
    and tested, not left to the parsing library's implicit default; a
    vendor's rc/dev build must be *inside* the declared range to pass.
    """
    from packaging.specifiers import InvalidSpecifier, SpecifierSet

    try:
        return SpecifierSet(specifier, prereleases=True)
    except InvalidSpecifier as exc:
        raise ValueError(f"invalid specifier {specifier!r} for {distribution}: {exc}") from exc


def check_version_gate(gate: str) -> tuple[bool, str]:
    """(blocked, detail) for one gate against the installed distribution.

    Unknown metadata does NOT block by itself (the patch's own source or
    capability probes decide); an out-of-range version always blocks.
    """
    name, specifier = parse_version_gate(gate)
    version = distribution_version(name)
    if version is None:
        return False, f"{name} metadata not found (patch probes decide)"
    try:
        spec = _specifier_set(name, specifier)
    except ValueError as exc:
        return True, str(exc)
    if spec.contains(version, prereleases=True):
        return False, f"{name} {version}"
    return True, f"{name} {version} does not match {specifier!r}"
