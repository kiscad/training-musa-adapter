"""Configuration: env switches, explicit TOML, freeze.

Sources, lowest to highest priority::

    built-in defaults  <  explicit TOML file  <  environment variables

Within one source, patch-specific options win over operator-generic ones.
``install(config_path=...)`` selects the file explicitly and wins over
``TRAINING_MUSA_ADAPTOR_CONFIG``.  The master switch
(``TRAINING_MUSA_ADAPTOR_ENABLED=0``) is a hard exit that no file or API
call can override.

Bootstrap reads only ENABLED/AUTOLOAD; everything else is parsed at the
first relevant import boundary and frozen there.  Unknown fields, patch
IDs, implementation names, enums or types raise :class:`ConfigError` --
nothing silently falls back to auto.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ._errors import ConfigError

__all__ = ["ENV_PREFIX", "Config", "ConfigManager"]

ENV_PREFIX = "TRAINING_MUSA_ADAPTOR"

ATTENTION_PATCH_ID = "megatron.te.attention.capability-dispatch"
ATTENTION_IMPLEMENTATIONS = frozenset(
    {"mudnn", "mate", "flash_attn", "te_unfused", "torch_sdpa_math"}
)
PATCH_OPTION_FIELDS = {
    ATTENTION_PATCH_ID: frozenset({"policy", "implementations", "fallback"})
}

_TRUTHY = {"1", "true"}
_FALSY = {"0", "false"}

_KNOWN_ENV = {
    "ENABLED",
    "AUTOLOAD",
    "CONFIG",
    "ONLY",
    "DISABLE",
    "ATTN_POLICY",
    "ATTN_IMPLS",
    "ATTN_FALLBACK",
    "DEBUG",
}

_POLICIES = ("auto", "prefer", "force", "upstream")
_FALLBACKS = ("reference", "upstream", "error")


def _parse_bool(raw: str, name: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSY:
        return False
    raise ConfigError(
        f"{ENV_PREFIX}_{name}={raw!r} is not a boolean; use 0/1/true/false"
    )


def _parse_id_list(raw: str, name: str) -> tuple[str, ...]:
    """Comma-separated list: whitespace trimmed; empty string means an
    empty list; duplicates, empty middle items and commas inside an ID or
    implementation name are errors."""
    if raw.strip() == "":
        return ()
    parts = [part.strip() for part in raw.split(",")]
    if any(not part for part in parts):
        raise ConfigError(f"{ENV_PREFIX}_{name}={raw!r} contains an empty item")
    if len(set(parts)) != len(parts):
        raise ConfigError(f"{ENV_PREFIX}_{name}={raw!r} contains duplicates")
    return tuple(parts)


@dataclass(frozen=True)
class Selection:
    """policy/implementations/fallback for one multi-implementation call site."""

    policy: str = "auto"
    implementations: tuple[str, ...] = ()
    fallback: str = "reference"

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "implementations": list(self.implementations),
            "fallback": self.fallback,
        }


@dataclass(frozen=True)
class Config:
    """Frozen effective configuration."""

    enabled: bool
    autoload: bool
    debug: bool
    only: tuple[str, ...]
    disable: tuple[str, ...]
    #: suite names as configured (before expansion into only/disable)
    only_suites: tuple[str, ...]
    disable_suites: tuple[str, ...]
    #: operator key ("attention") -> Selection
    operators: Mapping[str, Selection]
    #: patch id -> Selection (patch_options, only for declaring patches)
    patch_options: Mapping[str, Selection]
    config_path: str | None
    #: field path -> source ("defaults" | "file" | "env")
    sources: Mapping[str, str]
    #: human-readable notes (priority decisions, normalizations)
    notes: tuple[str, ...]

    def patch_enabled(self, patch_id: str, registered: set[str]) -> bool:
        """ONLY/DISABLE selection rules:

        - ONLY non-empty: whitelist; DISABLE is ignored entirely.
        - ONLY empty: the default set minus DISABLE.
        """
        if self.only:
            return patch_id in self.only
        return patch_id not in self.disable

    def attention(self) -> Selection:
        return self.operators["attention"]

    def options_for(self, patch_id: str) -> Selection | None:
        return self.patch_options.get(patch_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "autoload": self.autoload,
            "debug": self.debug,
            "patches": {
                "only": list(self.only),
                "disable": list(self.disable),
                "only_suites": list(self.only_suites),
                "disable_suites": list(self.disable_suites),
            },
            "operators": {
                key: value.as_dict() for key, value in self.operators.items()
            },
            "patch_options": {
                key: value.as_dict() for key, value in self.patch_options.items()
            },
            "config_path": self.config_path,
            "sources": dict(sorted(self.sources.items())),
            "notes": list(self.notes),
        }


#: Operator keys with a declared configuration section.
_OPERATOR_KEYS = ("attention",)


def _load_toml(path: str) -> dict[str, Any]:
    import sys

    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - py3.10 path
        import tomli as tomllib

    try:
        with open(path, "rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError:
        raise ConfigError(f"config file {path!r} does not exist") from None
    except Exception as exc:  # noqa: BLE001 - TOML syntax errors are config errors
        raise ConfigError(f"invalid TOML in {path!r}: {exc}") from exc


def _selection_from(
    section: Mapping[str, Any],
    *,
    base: Selection,
    context: str,
    available_impls: frozenset[str],
    sources: dict[str, str],
    source: str,
) -> Selection:
    if not isinstance(section, dict):
        raise ConfigError(f"{context} must be a TOML table")
    unknown = set(section) - {"policy", "implementations", "fallback"}
    if unknown:
        raise ConfigError(f"{context} has unknown field(s): {sorted(unknown)}")
    policy = base.policy
    implementations = base.implementations
    fallback = base.fallback
    if "policy" in section:
        policy = section["policy"]
        if not isinstance(policy, str) or policy not in _POLICIES:
            raise ConfigError(f"{context}.policy {policy!r} must be one of {_POLICIES}")
        sources[f"{context}.policy"] = source
    if "implementations" in section:
        raw = section["implementations"]
        if not isinstance(raw, (list, tuple)) or any(
            not isinstance(item, str) or not item or "," in item for item in raw
        ):
            raise ConfigError(f"{context}.implementations must be a list of names")
        implementations = tuple(raw)
        if len(set(implementations)) != len(implementations):
            raise ConfigError(f"{context}.implementations must not repeat")
        for item in implementations:
            if item not in available_impls:
                raise ConfigError(
                    f"{context}: unknown implementation {item!r} "
                    f"(available: {sorted(available_impls)})"
                )
        sources[f"{context}.implementations"] = source
    if "fallback" in section:
        fallback = section["fallback"]
        if not isinstance(fallback, str) or fallback not in _FALLBACKS:
            raise ConfigError(
                f"{context}.fallback {fallback!r} must be one of {_FALLBACKS}"
            )
        sources[f"{context}.fallback"] = source
    return Selection(policy, implementations, fallback)


def _validate_selection(selection: Selection, context: str) -> None:
    if selection.policy in ("auto", "upstream") and selection.implementations:
        raise ConfigError(
            f"{context}: policy {selection.policy!r} requires an empty implementations list"
        )
    if selection.policy == "force" and len(selection.implementations) != 1:
        raise ConfigError(
            f"{context}: policy 'force' requires exactly one implementation"
        )
    if selection.policy == "prefer" and not selection.implementations:
        raise ConfigError(
            f"{context}: policy 'prefer' requires a non-empty implementations list"
        )


def _normalize_fallback(
    selection: Selection, context: str, notes: list[str], sources: dict[str, str]
) -> Selection:
    """force/upstream always pair with fallback=error: other sources'
    fallback is ignored and the normalization is reported."""
    if selection.policy in ("force", "upstream") and selection.fallback != "error":
        notes.append(
            f"{context}: policy {selection.policy!r} normalizes fallback "
            f"{selection.fallback!r} to 'error'"
        )
        sources[f"{context}.fallback"] = "normalized"
        return Selection(selection.policy, selection.implementations, "error")
    return selection


class ConfigManager:
    """Deferred-parse, freeze-once configuration holder."""

    def __init__(self) -> None:
        self._config_path_override: str | None = None
        self._frozen = False
        self._config: Config | None = None

    # -- early bootstrap switches (stdlib only) -------------------------------

    def _early(self, name: str, default: bool) -> bool:
        raw = os.environ.get(f"{ENV_PREFIX}_{name}")
        if raw is None:
            return default
        return _parse_bool(raw, name)

    def enabled(self) -> bool:
        return self._early("ENABLED", True)

    def autoload_enabled(self) -> bool:
        return self.enabled() and self._early("AUTOLOAD", True)

    def debug(self) -> bool:
        if self._config is not None:
            return self._config.debug
        return self._early("DEBUG", False)

    # -- explicit file selection -------------------------------------------------

    def set_config_path(self, path: str) -> None:
        """``install(config_path=...)``; must happen before the freeze."""
        if self._frozen:
            raise ConfigError("configuration already frozen; start a new process")
        self._config_path_override = path

    # -- freeze ------------------------------------------------------------------

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def config(self) -> Config:
        if self._config is None:
            self.freeze()
        assert self._config is not None
        return self._config

    def freeze(
        self,
        *,
        registered_patch_ids: Iterable[str] = (),
        patch_option_declarations: Mapping[str, frozenset[str]] | None = None,
        available_impls: Mapping[str, frozenset[str]] | None = None,
        patch_suites: Mapping[str, str] | None = None,
    ) -> Config:
        """Parse all sources once and freeze.

        ``patch_option_declarations`` maps patch id -> the option fields the
        patch declares (``policy``/``implementations``/``fallback``);
        ``available_impls`` maps operator key -> declared implementation
        names.  Both come from the patch set at the first boundary.
        ``patch_suites`` maps patch id -> suite name; ONLY/DISABLE entries
        naming a suite expand to every patch id of that suite.
        """
        if self._config is not None:
            return self._config
        config = self._build(
            set(registered_patch_ids),
            (
                PATCH_OPTION_FIELDS
                if patch_option_declarations is None
                else patch_option_declarations
            ),
            (
                {"attention": ATTENTION_IMPLEMENTATIONS}
                if available_impls is None
                else available_impls
            ),
            patch_suites,
        )
        self._config = config
        self._frozen = True
        return config

    def _build(
        self,
        registered_patch_ids: set[str],
        patch_option_declarations: Mapping[str, frozenset[str]],
        available_impls: Mapping[str, frozenset[str]],
        patch_suites: Mapping[str, str] | None,
    ) -> Config:
        notes: list[str] = []
        sources: dict[str, str] = {}
        enabled = True
        autoload = True
        debug = False
        only: tuple[str, ...] = ()
        disable: tuple[str, ...] = ()
        operators: dict[str, Selection] = {key: Selection() for key in _OPERATOR_KEYS}
        patch_options: dict[str, Selection] = {}

        attn_impls = available_impls.get("attention", frozenset())

        # ---- TOML file (explicit path beats the env pointer) ----------------
        config_path = self._config_path_override or os.environ.get(
            f"{ENV_PREFIX}_CONFIG"
        )
        if config_path:
            data = _load_toml(config_path)
            unknown = set(data) - {"patches", "attention", "patch_options"}
            if unknown:
                raise ConfigError(
                    f"{config_path}: unknown top-level section(s): {sorted(unknown)}"
                )
            patches_section = data.get("patches", {})
            if not isinstance(patches_section, dict):
                raise ConfigError(f"{config_path}: patches must be a TOML table")
            unknown = set(patches_section) - {"only", "disable"}
            if unknown:
                raise ConfigError(
                    f"{config_path}: [patches] unknown field(s): {sorted(unknown)}"
                )
            for key in ("only", "disable"):
                if key in patches_section:
                    value = patches_section[key]
                    if not isinstance(value, list) or any(
                        not isinstance(item, str) or not item or "," in item
                        for item in value
                    ):
                        raise ConfigError(
                            f"{config_path}: patches.{key} must be a list of patch ids"
                        )
                    if len(set(value)) != len(value):
                        raise ConfigError(
                            f"{config_path}: patches.{key} contains duplicates"
                        )
                    if key == "only":
                        only = tuple(value)
                    else:
                        disable = tuple(value)
                    sources[f"patches.{key}"] = "file"
            if "attention" in data:
                operators["attention"] = _selection_from(
                    data["attention"],
                    base=operators["attention"],
                    context="attention",
                    available_impls=attn_impls,
                    sources=sources,
                    source="file",
                )
            options = data.get("patch_options", {})
            if not isinstance(options, dict):
                raise ConfigError(f"{config_path}: patch_options must be a TOML table")
            for patch_id, section in options.items():
                if patch_id not in registered_patch_ids:
                    raise ConfigError(
                        f"{config_path}: patch_options for unknown patch id {patch_id!r}"
                    )
                if not isinstance(section, dict):
                    raise ConfigError(
                        f"patch_options.{patch_id!r} must be a TOML table"
                    )
                declared = patch_option_declarations.get(patch_id, frozenset())
                unknown = set(section) - declared
                if unknown:
                    raise ConfigError(
                        f"{config_path}: patch_options.{patch_id!r} fields "
                        f"{sorted(unknown)} are not declared by the patch "
                        f"(declared: {sorted(declared)})"
                    )
                context = f'patch_options."{patch_id}"'
                for field in ("policy", "implementations", "fallback"):
                    sources[f"{context}.{field}"] = sources.get(
                        f"attention.{field}", "defaults"
                    )
                patch_options[patch_id] = _selection_from(
                    section,
                    base=operators["attention"],
                    context=f'patch_options."{patch_id}"',
                    available_impls=attn_impls,
                    sources=sources,
                    source="file",
                )

        # ---- environment ----------------------------------------------------
        for name in list(os.environ):
            if not name.startswith(ENV_PREFIX + "_"):
                continue
            suffix = name[len(ENV_PREFIX) + 1 :]
            if suffix not in _KNOWN_ENV:
                raise ConfigError(
                    f"unknown environment variable {name!r}; see README.md (Configuration / 配置)"
                )
        if (raw := os.environ.get(f"{ENV_PREFIX}_ENABLED")) is not None:
            enabled = _parse_bool(raw, "ENABLED")
            sources["enabled"] = "env"
        if (raw := os.environ.get(f"{ENV_PREFIX}_AUTOLOAD")) is not None:
            autoload = _parse_bool(raw, "AUTOLOAD")
            sources["autoload"] = "env"
        if (raw := os.environ.get(f"{ENV_PREFIX}_DEBUG")) is not None:
            debug = _parse_bool(raw, "DEBUG")
            sources["debug"] = "env"
        if (raw := os.environ.get(f"{ENV_PREFIX}_ONLY")) is not None:
            only = _parse_id_list(raw, "ONLY")
            sources["patches.only"] = "env"
        if (raw := os.environ.get(f"{ENV_PREFIX}_DISABLE")) is not None:
            disable = _parse_id_list(raw, "DISABLE")
            sources["patches.disable"] = "env"

        # ---- suite expansion ------------------------------------------------
        # ONLY/DISABLE entries may name a registered suite; such an entry
        # expands to every patch id of that suite (registry order, deduped).
        # Expansion happens after both sources are collected: environment
        # replaces the file's lists entirely, so the last source wins whole.
        only_suites: tuple[str, ...] = ()
        disable_suites: tuple[str, ...] = ()
        if patch_suites:
            suite_names = set(patch_suites.values())

            def _expand(
                entries: tuple[str, ...],
            ) -> tuple[tuple[str, ...], tuple[str, ...]]:
                ids: list[str] = []
                suites: list[str] = []
                for entry in entries:
                    if entry in suite_names:
                        suites.append(entry)
                        ids.extend(
                            pid for pid, suite in patch_suites.items() if suite == entry
                        )
                    else:
                        ids.append(entry)
                return tuple(dict.fromkeys(ids)), tuple(suites)

            only, only_suites = _expand(only)
            disable, disable_suites = _expand(disable)
            for suites, list_name in (
                (only_suites, "only"),
                (disable_suites, "disable"),
            ):
                if suites:
                    sources[f"patches.{list_name}_suites"] = sources.get(
                        f"patches.{list_name}", "defaults"
                    )
                    count = sum(1 for suite in patch_suites.values() if suite in suites)
                    notes.append(
                        f"suite(s) {sorted(suites)} expand to {count} patch id(s) "
                        f"in {list_name}"
                    )

        # ONLY/DISABLE validation and the old-project priority rule.
        unknown_ids = (set(only) | set(disable)) - registered_patch_ids
        if unknown_ids:
            raise ConfigError(
                f"unknown patch id(s) in ONLY/DISABLE: {sorted(unknown_ids)}; "
                "run `training-musa-adaptor list` for registered ids"
            )
        if only and disable:
            notes.append(
                "ONLY is non-empty: whitelist mode, DISABLE is ignored "
                "(old-project priority rule)"
            )

        attn_section: dict[str, Any] = {}
        if (raw := os.environ.get(f"{ENV_PREFIX}_ATTN_POLICY")) is not None:
            attn_section["policy"] = raw
        if (raw := os.environ.get(f"{ENV_PREFIX}_ATTN_IMPLS")) is not None:
            attn_section["implementations"] = list(_parse_id_list(raw, "ATTN_IMPLS"))
        if (raw := os.environ.get(f"{ENV_PREFIX}_ATTN_FALLBACK")) is not None:
            attn_section["fallback"] = raw
        if attn_section:
            operators["attention"] = _selection_from(
                attn_section,
                base=operators["attention"],
                context="attention",
                available_impls=attn_impls,
                sources=sources,
                source="env",
            )

        # Higher-priority generic environment values override file-specific
        # values too. Merge only supplied fields, before validating combinations.
        for patch_id, selection in patch_options.items():
            patch_options[patch_id] = _selection_from(
                attn_section,
                base=selection,
                context=f'patch_options."{patch_id}"',
                available_impls=attn_impls,
                sources=sources,
                source="env",
            )

        # ---- validation + normalization ------------------------------------
        _validate_selection(operators["attention"], "attention")
        operators["attention"] = _normalize_fallback(
            operators["attention"], "attention", notes, sources
        )
        for patch_id, selection in patch_options.items():
            _validate_selection(selection, f'patch_options."{patch_id}"')
            patch_options[patch_id] = _normalize_fallback(
                selection, f'patch_options."{patch_id}"', notes, sources
            )

        for field in ("enabled", "autoload", "debug"):
            sources.setdefault(field, "defaults")
        for key in ("only", "disable", "only_suites", "disable_suites"):
            sources.setdefault(f"patches.{key}", "defaults")
        for field in ("policy", "implementations", "fallback"):
            sources.setdefault(f"attention.{field}", "defaults")

        return Config(
            enabled=enabled,
            autoload=autoload,
            debug=debug,
            only=only,
            disable=disable,
            only_suites=only_suites,
            disable_suites=disable_suites,
            operators=MappingProxyType(operators),
            patch_options=MappingProxyType(patch_options),
            config_path=config_path,
            sources=MappingProxyType(sources),
            notes=tuple(notes),
        )

    def reset_for_tests(self) -> None:
        """Test-only: unfreeze (never available on the hot path)."""
        self._frozen = False
        self._config = None
        self._config_path_override = None
