"""Legacy configuration migrator (design doc §13.3, task T08).

Reads the old ``MEGATRON_MUSA_PATCH*`` environment (either the live process
environment or a ``KEY=VALUE`` dump file produced by ``env | grep``), emits a
musa-adapter TOML and a *diff report* that names every non-equivalence.

The old variables are never read by the new runtime; this tool exists so the
migration is explicit and reviewable::

    python tools/migrate_legacy_config.py --dump old-env.txt --out musa-adapter.toml
    python tools/migrate_legacy_config.py --live --out musa-adapter.toml
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import IO

OLD_PREFIX = "MEGATRON_MUSA_PATCH"

#: Old switch -> migration rule.  "conflict" entries cannot be converted
#: automatically and must be resolved by the operator.
_RULES = {
    "": ("runtime", "enabled", "exact"),
    "AUTOLOAD": ("runtime", "autoload", "exact"),
    "ONLY": ("patches", "patches.only", "json-list"),
    "DISABLE": ("patches", "patches.disable", "json-list"),
    "ATTN_BACKEND": ("operators.attention", None, "attention-backend"),
    "QWEN3VL_RMS_NORM": ("operators.rms_norm", None, "qwen3vl-rms-norm"),
    "IGNORE_VERSION_GATES": (None, None, "conflict"),
    "DEBUG": (None, None, "hint"),
    "STRICT": (None, None, "hint"),
    "ARCH": (None, None, "hint"),
}


def _parse_value(raw: str):
    lowered = raw.strip().lower()
    if lowered in ("1", "true"):
        return True
    if lowered in ("0", "false", ""):
        return False
    return raw.strip()


def _toml_list(items: list[str]) -> str:
    return "[" + ", ".join(json.dumps(item) for item in items) + "]"


def migrate(env: dict[str, str]) -> tuple[list[str], list[str]]:
    """Returns (toml_lines, report_lines)."""
    toml: list[str] = []
    report: list[str] = []
    attention_section: dict[str, str] = {}
    rms_section: dict[str, str] = {}

    runtime_section: dict[str, bool] = {}

    for key, value in sorted(env.items()):
        if not key.startswith(OLD_PREFIX):
            continue
        suffix = key[len(OLD_PREFIX) :].lstrip("_")
        rule = _RULES.get(suffix)
        if rule is None:
            report.append(f"UNMAPPED old variable {key}={value!r}: no migration rule")
            continue
        kind, target, strategy = rule
        if strategy == "exact":
            parsed = _parse_value(value)
            if parsed is False:
                runtime_section[target] = False
                report.append(f"{key}={value!r} -> runtime.{target}=false (exact)")
            else:
                report.append(f"{key}={value!r} -> default (already true)")
            continue
        if strategy == "json-list":
            report.append(
                f"{key}={value!r} -> {target}: IDs must be remapped via "
                "docs/MIGRATION_LEDGER.md; DISABLE now wins over ONLY"
            )
            continue
        if strategy == "attention-backend":
            if value == "auto":
                report.append(f"{key}=auto -> operators.attention policy=auto (default order is versioned, not path-identical)")
            elif value == "mate":
                attention_section["policy"] = "prefer"
                attention_section["providers"] = _toml_list(["attention.mate"])
                report.append(
                    f"{key}=mate -> prefer + attention.mate (old value was a "
                    "preference; NOT converted to force)"
                )
            elif value == "unfused":
                attention_section["policy"] = "force"
                attention_section["providers"] = _toml_list(["attention.te_unfused"])
                attention_section["fallback"] = "error"
                report.append(
                    f"{key}=unfused -> force + attention.te_unfused (only valid "
                    "for bindings that offer this provider)"
                )
            elif value == "mudnn":
                report.append(
                    f"{key}=mudnn: NO automatic conversion. The old value only "
                    "gated the forward; the new attention.mudnn checks the "
                    "backward window too. Choose prefer or force explicitly."
                )
            else:
                report.append(f"{key}={value!r}: unknown old backend value")
            continue
        if strategy == "qwen3vl-rms-norm":
            enabled = _parse_value(value)
            if enabled:
                report.append(
                    f"{key}=1 -> default (rms_norm auto already prefers the "
                    "fused provider)"
                )
            else:
                rms_section["policy"] = "upstream"
                report.append(f"{key}=0 -> operators.rms_norm policy=upstream")
            continue
        if strategy == "conflict":
            report.append(
                f"{key}={value!r}: NOT converted. Re-declare the experimental "
                "scope with evidence; the new runtime has no global "
                "ignore-all-version-gates switch."
            )
            continue
        if strategy == "hint":
            report.append(
                f"{key}={value!r}: no direct equivalent; see "
                "`musa-adapter doctor` and MUSA_ADAPTER_DIAGNOSTICS__LEVEL"
            )

    lines: list[str] = ["schema_version = 1", ""]
    if runtime_section:
        lines.append("[runtime]")
        for name, value in runtime_section.items():
            lines.append(f"{name} = {str(value).lower()}")
        lines.append("")
    if attention_section:
        lines.append("[operators.attention]")
        lines.append(f'policy = "{attention_section.get("policy", "auto")}"')
        if "providers" in attention_section:
            lines.append(f"providers = {attention_section['providers']}")
        if "fallback" in attention_section:
            lines.append(f'fallback = "{attention_section["fallback"]}"')
        lines.append("")
    if rms_section:
        lines.append("[operators.rms_norm]")
        lines.append(f'policy = "{rms_section["policy"]}"')
        lines.append("")
    if len(lines) == 2:
        lines.append("# (no convertible settings found)")
    return lines, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--live", action="store_true", help="read os.environ")
    source.add_argument("--dump", metavar="FILE", help="read a KEY=VALUE dump")
    parser.add_argument("--out", metavar="FILE", required=True)
    parser.add_argument("--report", metavar="FILE", default=None)
    args = parser.parse_args(argv)

    if args.live:
        import os

        env = dict(os.environ)
    else:
        env = {}
        with open(args.dump, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip().strip('"').strip("'")

    toml_lines, report_lines = migrate(env)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(toml_lines) + "\n")
    report_text = (
        "musa-adapter legacy-config migration report\n"
        "===========================================\n"
        "Review every line below before adopting the generated TOML.\n"
        "Semantics marked NOT equivalent must be re-validated.\n\n"
        + "\n".join(report_lines)
        + "\n"
    )
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(report_text)
    else:
        sys.stderr.write(report_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
