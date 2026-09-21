"""Small, lazy patch registry with per-target transactions and explicit undo.

Ported from megatron-musa-patch's engine with the v2.0 design changes:

- no framework-specific checks (multi-framework by construction);
- hooks run at the real ``exec_module`` boundary (see ``_imports.py``),
  never inside ``find_spec``;
- ONLY/DISABLE comes from the frozen config (old-project priority rule);
- rebind_prefixes defaults to empty -- alias repair is an explicit,
  scoped migration choice;
- installing while a known legacy engine is active is refused.

Attribute factories compose in PATCHES order: A listed before B yields
``B(A(original))``.  A target chain is committed only when every factory
succeeds; repeat application does not stack wrappers.  Hooks run before
their trigger module executes and own their cleanup; a hook whose boundary
was missed is reported phase_missed, never silently re-run late.  No
upstream source is rewritten.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import sys
import types
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Union

from . import _config as _config_mod
from ._compat import (
    check_version_gate,
    require_attr,
    split_target,
    validate_version_gates,
)
from ._errors import (
    ConfigError,
    EngineOverlapError,
    PatchConflict,
    PatchTargetMissing,
    TrainingMusaAdaptorError,
)
from ._imports import ImportWatcher

__all__ = ["AttrPatch", "HookPatch", "AppliedPatch", "Engine", "Patch"]

logger = logging.getLogger("training_musa_adaptor")

_PATCH_STATUSES = ("pending", "applied", "skipped", "failed", "reverted")


@dataclass(frozen=True)
class AttrPatch:
    """Replace a ``module:attribute`` (including ``module:Class.method``).

    ``replace(current)`` returns a replacement, or None to decline.  The
    factory only builds the object; the engine resolves, writes, makes it
    idempotent and undoable.  Declining, being disabled by configuration,
    and failing a version gate are three distinct report states -- None
    must never swallow an exception.

    ``requires`` names companion AttrPatches on *other attributes of the
    same module*; companions run first regardless of registration order.
    Missing, disabled or declined companions leave the consumer skipped
    with a reason; filtering never implicitly enables another patch.
    Cycles, same-target and cross-module requires are rejected.
    """

    id: str
    target: str
    replace: Callable[[Any], Any]
    rationale: str = ""
    upstream: str = ""
    remove_when: str = ""
    strategy: str = ""
    rebind_prefixes: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    # Declarative applicability gates, e.g. ("transformer_engine >=2.0,<2.1",).
    # Evaluated lazily (packaging, prereleases explicitly included) at the
    # patch boundary; a blocked gate skips the patch with the reason.
    version_gates: tuple[str, ...] = ()

    @property
    def module_name(self) -> str:
        return split_target(self.target)[0]

    @property
    def attr_name(self) -> str:
        return split_target(self.target)[1]

    @property
    def attr_leaf(self) -> str:
        return self.attr_name.rpartition(".")[2]

    def __post_init__(self) -> None:
        split_target(self.target)
        if not callable(self.replace):
            raise TypeError("AttrPatch.replace must be callable")
        if not isinstance(self.rebind_prefixes, tuple) or any(
            not prefix or not isinstance(prefix, str) for prefix in self.rebind_prefixes
        ):
            raise ValueError("rebind_prefixes must be a tuple of nonempty module prefixes")
        if not isinstance(self.requires, tuple) or any(
            not isinstance(item, str) or not item or item == self.id for item in self.requires
        ):
            raise ValueError("requires must contain nonempty companion patch ids, not self")
        validate_version_gates(self.version_gates)


@dataclass(frozen=True)
class HookPatch:
    """Run before ``trigger``'s real exec_module; return False to decline.

    ``trigger`` must name a module with a real execution body that runs
    before the behavior the hook prepares (namespace roots are not hook
    points).  ``undo`` cleans up owned changes.  A hook without undo is
    one-shot even across uninstall/reinstall and stays reported applied:
    irreversible external effects must not be described as restored.  A
    failing hook cleans up its own partial mutations before the exception
    propagates.
    """

    id: str
    trigger: str
    run: Callable[[], Any]
    rationale: str = ""
    upstream: str = ""
    remove_when: str = ""
    strategy: str = ""
    undo: Callable[[], None] | None = None
    version_gates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.trigger or not all(part.isidentifier() for part in self.trigger.split(".")):
            raise ValueError("HookPatch.trigger must name a module")
        if not callable(self.run):
            raise TypeError("HookPatch.run must be callable")
        if self.undo is not None and not callable(self.undo):
            raise TypeError("HookPatch.undo must be callable")
        validate_version_gates(self.version_gates)


Patch = Union[AttrPatch, HookPatch]


@dataclass
class AppliedPatch:
    """Public diagnostic record; runtime bindings are kept separately."""

    patch: Patch
    status: str = "pending"  # pending | applied | skipped | failed | reverted
    detail: str = ""

    def as_dict(self) -> dict:
        patch = self.patch
        return {
            "id": patch.id,
            "kind": "attr" if isinstance(patch, AttrPatch) else "hook",
            "target": patch.target if isinstance(patch, AttrPatch) else f"{patch.trigger} (hook)",
            "version_gates": list(patch.version_gates),
            "status": self.status,
            "detail": self.detail,
            "rationale": patch.rationale,
            "strategy": patch.strategy,
            "upstream": patch.upstream,
            "remove_when": patch.remove_when,
            "requires": list(patch.requires) if isinstance(patch, AttrPatch) else [],
        }


@dataclass
class _Binding:
    owner: Any
    leaf: str
    original: Any  # raw descriptor, not its bound value
    replacement: Any
    owned: bool  # inherited attributes must be deleted on undo
    rebind_prefixes: tuple[str, ...] = ()
    aliases: list[tuple[types.ModuleType, str, Any]] = field(default_factory=list)

    def restore(self, targets: set) -> None:
        if inspect.getattr_static(self.owner, self.leaf, None) is self.replacement:
            if self.owned:
                setattr(self.owner, self.leaf, self.original)
            else:
                delattr(self.owner, self.leaf)
        else:
            logger.warning("not restoring %s: another writer replaced the patch", self.leaf)
        for module, name, original in reversed(self.aliases):
            if vars(module).get(name) is self.replacement:
                setattr(module, name, original)
        # Also repair same-name consumers imported after initial application.
        _rebind_aliases(
            self.owner, self.leaf, self.replacement, self.original, self.rebind_prefixes, targets
        )


class Engine:
    """Registry + import watcher.  Configure before imports, not during training.

    Each target chain is atomic; arbitrary hook effects or imports are not a
    process-wide transaction.  On failure report() retains diagnostics and
    uninstall() can undo successful, owned changes.
    """

    def __init__(self) -> None:
        self._records: dict[str, AppliedPatch] = {}
        self._attrs: dict[str, list[AttrPatch]] = {}
        self._hooks: dict[str, list[HookPatch]] = {}
        self._bindings: dict[tuple[str, str], _Binding] = {}
        self._undo_order: list[Union[tuple[str, str], HookPatch]] = []
        self._running_hooks: set[str] = set()
        self._watcher = ImportWatcher(self)
        self._installed = False
        self._config_ready = False
        self._applied_sweep_done = False
        self._restart_required: list[str] = []
        #: initialization failure saved by the automatic channel; raised at
        #: the first relevant boundary (design doc §5.5)
        self._saved_error: Exception | None = None
        self.config = _config_mod.ConfigManager()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, patches: Iterable[Patch]) -> None:
        """Validate a whole batch, then register it (also works after install)."""
        patches = tuple(patches)
        ids = set(self._records)
        for patch in patches:
            if not isinstance(patch, (AttrPatch, HookPatch)):
                raise TypeError("patches must be AttrPatch or HookPatch instances")
            if not patch.id:
                raise ValueError("patch id must not be empty")
            if patch.id in ids:
                raise ValueError(f"duplicate patch id: {patch.id!r}")
            ids.add(patch.id)
        self._validate_dependencies(
            [record.patch for record in self._records.values()] + list(patches)
        )
        for patch in patches:
            self._records[patch.id] = AppliedPatch(patch)
            if self._config_ready:
                self._schedule(patch)
        if self._installed and self._config_ready:
            self._apply_already_imported()

    @staticmethod
    def _validate_dependencies(patches: list[Patch]) -> None:
        """Validate target-level DAGs without importing dependency modules.

        requires only names companions on other attributes of the same
        module; cycles, same-target and cross-module requires are rejected
        (design doc §5.3).
        """
        records = {patch.id: patch for patch in patches}
        edges: dict[tuple[str, str], set[tuple[str, str]]] = {}
        for patch in patches:
            if not isinstance(patch, AttrPatch):
                continue
            key = (patch.module_name, patch.attr_name)
            for pid in patch.requires:
                dependency = records.get(pid)
                if dependency is None:
                    continue
                if (
                    not isinstance(dependency, AttrPatch)
                    or dependency.module_name != patch.module_name
                    or dependency.attr_name == patch.attr_name
                ):
                    raise ValueError(
                        f"{patch.id}: requires {pid} must target another attribute "
                        "in the same module"
                    )
                edges.setdefault(key, set()).add((dependency.module_name, dependency.attr_name))
        visited, visiting = set(), set()

        def visit(key):
            if key in visiting:
                raise ValueError(f"cyclic patch requirements at {key}")
            if key in visited:
                return
            visiting.add(key)
            for dependency in edges.get(key, ()):
                visit(dependency)
            visiting.remove(key)
            visited.add(key)

        for key in edges:
            visit(key)

    # ------------------------------------------------------------------
    # Configuration & scheduling
    # ------------------------------------------------------------------

    def _declared_option_fields(self) -> dict[str, frozenset[str]]:
        """Patches may declare patch_options fields; V1: attention patch only."""
        return {
            "megatron.te.attention.capability-dispatch": frozenset(
                {"policy", "implementations", "fallback"}
            )
        }

    def _available_impls(self) -> dict[str, frozenset[str]]:
        from .ops import attention as _attention_ops

        return {"attention": frozenset(_attention_ops.DECLARED_IMPLEMENTATIONS)}

    def _ensure_config(self) -> None:
        """Parse + freeze configuration once, at the first relevant boundary."""
        if self._saved_error is not None:
            saved, self._saved_error = self._saved_error, None
            raise TrainingMusaAdaptorError(
                f"initialization failed earlier: {type(saved).__name__}: {saved}"
            ) from saved
        if self._config_ready or not self._installed:
            return
        config = self.config.freeze(
            registered_patch_ids=set(self._records),
            patch_option_declarations=self._declared_option_fields(),
            available_impls=self._available_impls(),
        )
        if not config.enabled:
            # Total switch from the file: no patches apply.
            for record in self._records.values():
                if record.status == "pending":
                    record.status, record.detail = "skipped", "disabled by configuration"
            self._config_ready = True
            return
        self._config_ready = True
        for record in self._records.values():
            self._schedule(record.patch)
        self._apply_already_imported()

    def _schedule(self, patch: Patch) -> None:
        record = self._records[patch.id]
        if record.status == "applied" and isinstance(patch, HookPatch):
            return  # irreversible hook retained across uninstall
        if record.status != "pending":
            return
        if not self.config.config.patch_enabled(patch.id, set(self._records)):
            record.status, record.detail = "skipped", "disabled by configuration"
        elif isinstance(patch, AttrPatch):
            self._attrs.setdefault(patch.module_name, []).append(patch)
        else:
            self._hooks.setdefault(patch.trigger, []).append(patch)

    # ------------------------------------------------------------------
    # Installation
    # ------------------------------------------------------------------

    def _check_engine_overlap(self) -> None:
        """Refuse to run alongside a known legacy patch engine (design §9.3)."""
        for marker, name in (
            ("__megatron_musa_patch_import_watcher__", "megatron-musa-patch"),
            ("__musa_adapter_import_watcher__", "musa-adapter (round 1)"),
        ):
            if any(getattr(finder, marker, False) for finder in sys.meta_path):
                raise EngineOverlapError(
                    f"{name} import watcher is already installed in this process; "
                    "two engines writing the same targets is not supported. "
                    "Disable the other engine's master switch or start a new process."
                )
        for module_name, name in (
            ("megatron_musa_patch", "megatron-musa-patch"),
            ("musa_adapter", "musa-adapter (round 1)"),
        ):
            module = sys.modules.get(module_name)
            if module is not None:
                raise EngineOverlapError(
                    f"{name} is already imported in this process; overlapping "
                    "patch engines are not supported. Start a new process."
                )

    def install(self, *, config_path: str | None = None, eager: bool = True) -> None:
        """Watch future imports and patch fully imported targets immediately.

        ``eager=False`` (the automatic torch.backends channel) only installs
        the watcher: configuration parsing is deferred to the first relevant
        import boundary so ``import torch`` stays light.
        """
        if not self.config.enabled():
            return
        if self._installed:
            if eager:
                self._ensure_config()
            return
        if self._bindings or any(
            isinstance(item, HookPatch) and self._records[item.id].status == "failed"
            for item in self._undo_order
        ):
            raise PatchConflict("patch cleanup is incomplete; retry uninstall() before install()")
        self._check_engine_overlap()
        if config_path is not None:
            self.config.set_config_path(config_path)
        self._installed = True
        try:
            sys.meta_path.insert(0, self._watcher)
        except Exception as exc:
            self._installed = False
            from ._errors import WatcherInstallError

            raise WatcherInstallError(
                f"cannot install the import watcher: {type(exc).__name__}: {exc}; "
                "there is no reliable interception point"
            ) from exc
        if eager:
            self._ensure_config()

    # ------------------------------------------------------------------
    # Import boundary callbacks (from _imports.py)
    # ------------------------------------------------------------------

    def _watches(self, fullname: str) -> bool:
        if not self._installed:
            return False
        if not self._config_ready:
            # Before the freeze, watch every module some patch points at;
            # disabled patches are filtered when the boundary fires.
            for record in self._records.values():
                patch = record.patch
                if isinstance(patch, AttrPatch) and patch.module_name == fullname:
                    return True
                if isinstance(patch, HookPatch) and patch.trigger == fullname:
                    return True
            return False
        return fullname in self._attrs or fullname in self._hooks

    def _mark_absent(self, fullname: str) -> None:
        patches = self._attrs.pop(fullname, []) + self._hooks.pop(fullname, [])
        for patch in patches:
            record = self._records[patch.id]
            record.status, record.detail = "skipped", f"module {fullname!r} is not installed"

    def _mark_namespace(self, fullname: str) -> None:
        """A namespace package has no execution body: hooks here are invalid,
        attr patches keep waiting for a concrete submodule boundary."""
        for patch in self._hooks.pop(fullname, []):
            record = self._records[patch.id]
            record.status = "failed"
            record.detail = (
                f"hook trigger {fullname!r} is a namespace package with no "
                "execution body; choose a concrete module boundary"
            )
        # attr patches on a namespace module can never resolve either, but a
        # submodule of the same prefix may carry the target; leave them.

    def _mark_import_failed(self, fullname: str, exc: Exception) -> None:
        for patch in self._attrs.get(fullname, ()):
            record = self._records[patch.id]
            if record.status == "pending":
                record.status, record.detail = (
                    "failed",
                    f"import failed: {type(exc).__name__}: {exc}",
                )

    def _run_hooks(self, module_name: str) -> None:
        """before-exec boundary: run this module's hooks (once)."""
        self._ensure_config()
        if module_name in self._running_hooks:
            return
        hooks = self._hooks.get(module_name)
        if not hooks:
            return
        self._running_hooks.add(module_name)
        try:
            for patch in hooks:
                record = self._records[patch.id]
                if record.status in {"applied", "skipped"}:
                    continue
                try:
                    gate_detail = self._version_gate_block(patch)
                    if gate_detail is not None:
                        record.status, record.detail = "skipped", f"version gate: {gate_detail}"
                        self._log(record)
                        continue
                    result = patch.run()
                except Exception as exc:
                    # A failing hook must have cleaned its own partial
                    # mutations; the engine records and propagates (§5.5).
                    record.status, record.detail = "failed", f"{type(exc).__name__}: {exc}"
                    raise
                record.status = "skipped" if result is False else "applied"
                record.detail = "hook declined to run" if result is False else "hook completed"
                if result is not False:
                    self._undo_order.append(patch)
                self._log(record)
            self._hooks.pop(module_name, None)
        finally:
            self._running_hooks.discard(module_name)

    def _apply_for_module(
        self, module_name: str, module: types.ModuleType, *, trigger: str
    ) -> None:
        """after-exec boundary: apply attr patches for one module."""
        self._ensure_config()
        groups: dict[str, list[AttrPatch]] = {}
        for patch in self._attrs.get(module_name, ()):
            groups.setdefault(patch.attr_name, []).append(patch)
        visited = set()

        def apply_target(attr):
            if attr in visited:
                return
            visited.add(attr)
            for patch in groups[attr]:
                for pid in patch.requires:
                    record = self._records.get(pid)
                    if (
                        record is not None
                        and isinstance(record.patch, AttrPatch)
                        and record.patch.attr_name in groups
                    ):
                        apply_target(record.patch.attr_name)
            self._apply_target(module, attr, groups[attr], trigger)

        for attr in groups:
            apply_target(attr)

    # ------------------------------------------------------------------
    # Late application for already-imported modules
    # ------------------------------------------------------------------

    def _apply_already_imported(self) -> None:
        if self._applied_sweep_done:
            return
        self._applied_sweep_done = True
        for module_name in list(self._hooks):
            if sys.modules.get(module_name) is not None:
                # The import boundary is gone; never re-run a hook late.
                for patch in self._hooks.pop(module_name):
                    record = self._records[patch.id]
                    if record.status == "pending":
                        record.status = "skipped"
                        record.detail = (
                            f"phase_missed: {module_name!r} was already imported "
                            "before activation; start a new process for this hook"
                        )
                        self._log(record)
        for module_name in list(self._attrs):
            module = sys.modules.get(module_name)
            if module is not None and not getattr(
                getattr(module, "__spec__", None), "_initializing", False
            ):
                self._apply_for_module(module_name, module, trigger="already-imported")

    # ------------------------------------------------------------------
    # Gates and skip reasons
    # ------------------------------------------------------------------

    @staticmethod
    def _version_gate_block(patch: Patch) -> str | None:
        for gate in patch.version_gates:
            blocked, detail = check_version_gate(gate)
            if blocked:
                return detail
        return None

    def _declared_targets(self) -> set[tuple[str, str]]:
        return {
            (patch.module_name, patch.attr_name)
            for record in self._records.values()
            if isinstance(patch := record.patch, AttrPatch)
        }

    def _skip_reason(self, patch: AttrPatch) -> str | None:
        gate = self._version_gate_block(patch)
        if gate is not None:
            return f"version gate: {gate}"
        unavailable = [pid for pid in patch.requires if not self.is_applied(pid)]
        if unavailable:
            return "requires applied companion(s): " + ", ".join(unavailable)
        return None

    # ------------------------------------------------------------------
    # Target application (ported transaction)
    # ------------------------------------------------------------------

    def _apply_target(self, module, attr: str, patches: list[AttrPatch], trigger: str) -> None:
        key = (module.__name__, attr)
        previous = self._bindings.get(key)
        active = patches[0]
        mutations = []
        try:
            reasons = {p.id: self._skip_reason(p) for p in patches}
            # Ineligible consumers need not expose their target symbol. Resolve
            # only when a factory can run or an existing binding needs cleanup.
            if previous is None and all(detail is not None for detail in reasons.values()):
                for patch in patches:
                    record = self._records[patch.id]
                    record.status, record.detail = "skipped", reasons[patch.id] or ""
                    self._log(record)
                return
            owner_path, _, leaf = attr.rpartition(".")
            owner = require_attr(module, owner_path, patch_id=active.id, target=active.target) if owner_path else module
            require_attr(module, attr, patch_id=active.id, target=active.target)
            raw = inspect.getattr_static(owner, leaf)
            if previous is not None and previous.owner is owner and raw is previous.replacement:
                if all(
                    self._records[p.id].status in {"applied", "skipped"} for p in patches
                ) and not any(
                    self._records[p.id].detail.startswith("requires ")
                    and all(self.is_applied(pid) for pid in p.requires)
                    for p in patches
                ):
                    return
                # Newly registered patch on an existing target: rebuild the
                # whole chain from the baseline, not from an already wrapped fn.
                current = previous.original
            else:
                if previous is not None and trigger != "import":
                    raise PatchConflict(
                        f"patch target {key!r} changed outside the engine; uninstall first"
                    )
                current = raw
            baseline = current
            outcomes = []
            for active in patches:
                reason = reasons[active.id]
                if reason is not None:
                    outcomes.append((active, "skipped", reason))
                    continue
                # Preserve binding semantics for staticmethod/classmethod.
                value = (
                    current.__func__
                    if isinstance(current, (staticmethod, classmethod))
                    else current
                )
                replacement = active.replace(value)
                if replacement is None:
                    outcomes.append((active, "skipped", "patch declined to run"))
                    continue
                if isinstance(current, (staticmethod, classmethod)) and not isinstance(
                    replacement, (staticmethod, classmethod)
                ):
                    replacement = type(current)(replacement)
                current = replacement
                outcomes.append((active, "applied", f"trigger={trigger}"))
            if any(status == "applied" for _, status, _ in outcomes):
                binding = _Binding(owner, leaf, baseline, current, leaf in vars(owner))
                mutations.append((owner, leaf, raw, leaf in vars(owner), current))
                setattr(owner, leaf, current)
                if previous is not None:
                    # Existing consumer aliases may still point to the previous
                    # generation after reload. Keep their original undo values.
                    for consumer, name, _original in previous.aliases:
                        if vars(consumer).get(name) is previous.replacement:
                            mutations.append((consumer, name, previous.replacement, True, current))
                            setattr(consumer, name, current)
                            binding.aliases.append((consumer, name, baseline))
                    binding.owned = previous.owned if previous.owner is owner else binding.owned
                if not owner_path:
                    prefixes = tuple(
                        dict.fromkeys(prefix for p in patches for prefix in p.rebind_prefixes)
                    )
                    # An explicitly registered target has its own lifecycle.
                    # Rebinding it as a consumer would invalidate its ownership.
                    binding.rebind_prefixes = prefixes
                    declared = self._declared_targets()
                    sources = [raw]
                    if previous is not None and previous.replacement is not raw:
                        sources.append(previous.replacement)
                    for source in sources:
                        aliases = _rebind_aliases(module, leaf, source, current, prefixes, declared)
                        for consumer, name, old in aliases:
                            mutations.append((consumer, name, old, True, current))
                            binding.aliases.append((consumer, name, baseline))
                self._bindings[key] = binding
                if key not in self._undo_order:
                    self._undo_order.append(key)
            elif previous is not None:
                # Reload may make a conditional patch unnecessary. Consumers
                # holding the old replacement must follow the new upstream
                # baseline rather than keeping a retired wrapper indefinitely.
                for consumer, name, _ in previous.aliases:
                    if vars(consumer).get(name) is previous.replacement:
                        mutations.append((consumer, name, previous.replacement, True, baseline))
                        setattr(consumer, name, baseline)
                declared = self._declared_targets()
                _rebind_aliases(
                    owner, leaf, previous.replacement, baseline, previous.rebind_prefixes, declared
                )
                self._bindings.pop(key)
                self._undo_order.remove(key)
            for patch, status, detail in outcomes:
                record = self._records[patch.id]
                record.status, record.detail = status, detail
                self._log(record)
        except Exception as exc:
            for changed_owner, name, old, owned, replacement in reversed(mutations):
                if inspect.getattr_static(changed_owner, name, None) is replacement:
                    if owned:
                        setattr(changed_owner, name, old)
                    else:
                        delattr(changed_owner, name)
            record = self._records[active.id]
            record.status, record.detail = "failed", f"{type(exc).__name__}: {exc}"
            if isinstance(exc, TrainingMusaAdaptorError):
                raise
            raise TrainingMusaAdaptorError(
                f"patch {active.id!r} failed for {active.target!r} "
                f"(trigger={trigger}): {type(exc).__name__}: {exc}"
            ) from exc

    def _log(self, record: AppliedPatch) -> None:
        level = logging.INFO if self.config.debug() else logging.DEBUG
        logger.log(level, "%s: %s (%s)", record.patch.id, record.status, record.detail)

    # ------------------------------------------------------------------
    # Explicit apply (diagnostics)
    # ------------------------------------------------------------------

    def apply(self, patch_ids: Iterable[str]) -> None:
        """Import and apply only the named patches (diagnostic entry).

        Never mass-imports: patch_ids is required and validated.  Hooks whose
        boundary was already missed are reported phase_missed, not run late.
        """
        patch_ids = tuple(patch_ids)
        if not patch_ids:
            raise ValueError(
                "apply() requires explicit patch_ids; full-target import is not "
                "offered (design doc §7.3)"
            )
        if not self.config.enabled():
            return
        self.install()
        unknown = set(patch_ids) - set(self._records)
        if unknown:
            raise ConfigError(f"unknown patch id(s) for apply(): {sorted(unknown)}")
        self._ensure_config()
        modules: dict[str, list[Patch]] = {}
        for patch_id in patch_ids:
            patch = self._records[patch_id].patch
            name = patch.module_name if isinstance(patch, AttrPatch) else patch.trigger
            modules.setdefault(name, []).append(patch)
        for module_name in sorted(modules):
            already = sys.modules.get(module_name) is not None
            if not already:
                try:
                    module = importlib.import_module(module_name)
                except ModuleNotFoundError as exc:
                    if exc.name and (module_name == exc.name or module_name.startswith(exc.name + ".")):
                        self._mark_absent(module_name)
                        continue
                    self._mark_import_failed(module_name, exc)
                    raise
                except Exception as exc:
                    self._mark_import_failed(module_name, exc)
                    raise
                # The import boundary fired naturally during import_module.
                continue
            module = sys.modules[module_name]
            # Already imported: attr patches may still apply (late, caveated);
            # hooks are phase_missed and never re-run late (design §7.3).
            for patch in modules[module_name]:
                if isinstance(patch, HookPatch):
                    record = self._records[patch.id]
                    if record.status == "pending":
                        record.status = "skipped"
                        record.detail = (
                            f"phase_missed: {module_name!r} was already imported; "
                            "start a new process for this hook"
                        )
                        self._log(record)
            self._run_hooks(module_name)
            self._apply_for_module(module_name, module, trigger="apply")

    # ------------------------------------------------------------------
    # Uninstall & reporting
    # ------------------------------------------------------------------

    def uninstall(self) -> None:
        """Undo owned changes in reverse application order and clear watchers.

        Third-party writes are not overwritten.  Hooks without undo and other
        irreversible effects are reported as restart-required, not described
        as restored.
        """
        if self._watcher in sys.meta_path:
            sys.meta_path.remove(self._watcher)
        self._installed = False
        self._config_ready = False
        self._applied_sweep_done = False
        self._attrs.clear()
        self._hooks.clear()
        errors = []
        retained: list[Union[tuple[str, str], HookPatch]] = []
        targets = self._declared_targets()
        for item in reversed(self._undo_order):
            if isinstance(item, HookPatch):
                record = self._records[item.id]
                if item.undo is None:
                    record.detail = "hook has no undo; effects remain after uninstall"
                    if item.id not in self._restart_required:
                        self._restart_required.append(item.id)
                    retained.append(item)
                    continue
                try:
                    item.undo()
                except Exception as exc:
                    record.status, record.detail = "failed", f"undo failed: {exc}"
                    errors.append(exc)
                    retained.append(item)
            else:
                try:
                    self._bindings[item].restore(targets)
                except Exception as exc:
                    errors.append(exc)
                    retained.append(item)
                    for record in self._records.values():
                        owned = record.patch
                        if (
                            isinstance(owned, AttrPatch)
                            and (owned.module_name, owned.attr_name) == item
                        ):
                            record.status, record.detail = "failed", f"undo failed: {exc}"
                else:
                    self._bindings.pop(item)
        self._undo_order = list(reversed(retained))
        retained_ids = {item.id for item in retained if isinstance(item, HookPatch)}
        retained_ids.update(
            record.patch.id
            for record in self._records.values()
            if isinstance(record.patch, AttrPatch)
            and (record.patch.module_name, record.patch.attr_name) in retained
        )
        for record in self._records.values():
            if record.patch.id not in retained_ids and record.status != "skipped":
                if record.status in {"applied", "failed"}:
                    record.status, record.detail = "reverted", "uninstalled"
        if errors:
            raise TrainingMusaAdaptorError(
                f"patch cleanup failed: {errors[0]}; see report()"
            ) from errors[0]

    def is_applied(self, patch_id: str | None = None) -> bool:
        if patch_id is not None:
            record = self._records.get(patch_id)
            return bool(record and record.status == "applied")
        return any(record.status == "applied" for record in self._records.values())

    def report(self) -> dict:
        config_dict: dict[str, Any]
        if self._config_ready:
            config_dict = self.config.config.as_dict()
        else:
            config_dict = {"frozen": False}
        return {
            "patches": [record.as_dict() for record in self._records.values()],
            "config": config_dict,
            "restart_required": list(self._restart_required),
            "pending_modules": self.pending_modules(),
        }

    def pending_modules(self) -> list[str]:
        return sorted(
            {
                name
                for watch in (self._attrs, self._hooks)
                for name, patches in watch.items()
                if any(self._records[p.id].status == "pending" for p in patches)
            }
        )

    def __iter__(self) -> Iterator[AppliedPatch]:
        return iter(self._records.values())


def _rebind_aliases(module, attr: str, old: Any, new: Any, prefixes: tuple[str, ...], targets: set):
    """Bounded same-name alias repair; never scan attributes via __getattr__.

    Only modules under the explicitly declared prefixes are inspected
    (rebind_prefixes defaults to empty); primitive flags are never rebound.
    Transactional: a failure in one consumer rolls back the aliases already
    rebound, so the caller's binding stays the single source of truth.
    """
    if not prefixes:
        return []
    if not isinstance(old, (types.FunctionType, types.BuiltinFunctionType, type)):
        return []
    aliases: list[tuple[types.ModuleType, str, Any]] = []
    for name, consumer in list(sys.modules.items()):
        if (name, attr) in targets or consumer is module or type(consumer) is not types.ModuleType:
            continue
        if not any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            continue
        if attr in vars(consumer) and vars(consumer)[attr] is old:
            try:
                setattr(consumer, attr, new)
            except Exception:
                for rollback_module, rollback_attr, rollback_old in reversed(aliases):
                    try:
                        setattr(rollback_module, rollback_attr, rollback_old)
                    except Exception:  # noqa: BLE001 - best effort rollback
                        pass
                raise
            aliases.append((consumer, attr, old))
    return aliases
