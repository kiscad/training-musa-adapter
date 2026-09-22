"""Import watcher: one coordinated MetaPathFinder/loader wrapper.

Import-boundary contracts:

- ``find_spec`` only *locates and wraps* the spec.  Hook callbacks run at
  the loader's real ``exec_module`` boundary, so an external existence
  query (``importlib.util.find_spec("x")``) never applies a patch.
- The wrapper delegates the original loader's ``create_module``,
  ``exec_module`` and every other capability; nothing is re-implemented.
- Namespace packages have no execution body: they are not hook boundaries
  (the engine marks such hooks instead of guessing).
- Only watched targets are wrapped; unrelated modules are untouched.

Note: Python's own ``importlib.util.find_spec("a.b")`` may import the parent
package ``a`` itself -- this module only guarantees that *our* finder adds
no side effects on top of that platform behavior.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from typing import Any

__all__ = ["ImportWatcher", "WATCHER_MARKER", "find_spec_without_watchers"]

#: Marks our watchers on ``sys.meta_path`` so internal availability probes
#: can skip them: a plain existence query must not run pending actions.
WATCHER_MARKER = "__training_musa_adaptor_import_watcher__"

#: Exact markers of known external engines, not runtime dependencies.
#: Delegating find_spec
#: through each other would recurse; overlapping *writes* to the same
#: target are separately rejected by the engine.
_KNOWN_WATCHER_MARKERS = (
    WATCHER_MARKER,
    "__megatron_musa_patch_import_watcher__",
    "__musa_adapter_import_watcher__",
)


def find_spec_without_watchers(fullname: str, path=None, target=None):
    """find_spec delegating to every meta-path finder except import watchers."""
    for finder in sys.meta_path:
        if any(getattr(finder, marker, False) for marker in _KNOWN_WATCHER_MARKERS):
            continue
        find_spec = getattr(finder, "find_spec", None)
        if find_spec is None:
            continue
        spec = find_spec(fullname, path, target)
        if spec is not None:
            return spec
    return None


class _PatchedLoader(importlib.abc.Loader):
    """Wraps one loader; runs the engine's exec-boundary phases around it."""

    def __init__(self, wrapped: Any, fullname: str, engines: list[Any]) -> None:
        self._wrapped = wrapped
        self._fullname = fullname
        self._engines = engines

    def create_module(self, spec):
        create = getattr(self._wrapped, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module):
        # before-exec hooks (HookPatch.run) at the real boundary.
        for engine in self._engines:
            if engine._installed:
                engine._run_hooks(self._fullname)
        try:
            self._wrapped.exec_module(module)
        except Exception as exc:
            for engine in self._engines:
                engine._mark_import_failed(self._fullname, exc)
            raise
        # after-exec attr patches for this module.
        for engine in self._engines:
            if engine._installed:
                engine._apply_for_module(self._fullname, module, trigger="import")

    def __getattr__(self, item: str) -> Any:
        return getattr(self._wrapped, item)


class ImportWatcher(importlib.abc.MetaPathFinder):
    """One finder per engine; wraps loaders of watched trigger modules."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine
        setattr(self, WATCHER_MARKER, True)

    def find_spec(self, fullname, path=None, target=None):
        if not self._engine._watches(fullname):
            return None
        spec = find_spec_without_watchers(fullname, path, target)
        if spec is None:
            self._engine._mark_absent(fullname)
            return None
        if spec.loader is None:
            # Namespace package: no execution body to hook into.
            self._engine._mark_namespace(fullname)
            return spec
        loader = spec.loader
        # LazyLoader: unwrap so exec_module is really ours.  Only watched
        # targets get this treatment; unrelated lazy loaders stay lazy.
        if isinstance(loader, importlib.util.LazyLoader):
            inner = getattr(loader, "loader", None)
            loader = inner if inner is not None else loader
        # Coordinate all matching engines in installation order; skipping
        # sibling finders during lookup must not discard their callbacks.
        engines = [
            finder._engine
            for finder in reversed(sys.meta_path)
            if isinstance(finder, ImportWatcher) and finder._engine._watches(fullname)
        ]
        spec.loader = _PatchedLoader(loader, fullname, engines)
        return spec
