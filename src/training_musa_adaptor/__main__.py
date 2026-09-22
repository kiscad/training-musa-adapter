"""Small diagnostic CLI.

Read-only: ``list`` and ``config`` load declarations/configuration only,
``report`` describes the *current CLI process* -- it is never a running
training job's state.  Training-process reports come from the in-process
:func:`training_musa_adaptor.report` API.

Exit codes: 0 ok, 2 configuration/command error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import __version__
from ._errors import TrainingMusaAdaptorError

__all__ = ["main"]

_EXIT_OK = 0
_EXIT_CONFIG = 2


def _load_patches():
    from .patches import PATCHES

    return PATCHES


def _cmd_list(_args) -> int:
    for patch in _load_patches():
        kind = "attr" if hasattr(patch, "target") else "hook"
        target = getattr(patch, "target", None) or f"{patch.trigger} (hook)"
        print(f"{patch.id}\t{kind}\t{target}")
    return _EXIT_OK


def _cmd_config(_args) -> int:
    from ._config import ConfigManager
    from .patches import PATCH_SUITES

    manager = ConfigManager()
    # CLI config inspection freezes only this diagnostic process.
    config = manager.freeze(
        registered_patch_ids={patch.id for patch in _load_patches()},
        patch_suites=PATCH_SUITES,
    )
    print(json.dumps(config.as_dict(), indent=2))
    return _EXIT_OK


def _cmd_report(args) -> int:
    from . import report

    data = report()
    data["version"] = __version__
    if args.json:
        print(json.dumps(data, indent=2, default=str))
    else:
        print(f"training-musa-adaptor {__version__}")
        for patch in data["patches"]:
            print(f"  {patch['id']}: {patch['status']} ({patch['detail']})")
        print(
            "note: this report describes the CLI process; pending patches here "
            "are not a running training job's state"
        )
    return _EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="training-musa-adaptor",
        description="training-musa-adaptor diagnostics (read-only)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("list", help="list registered patches")
    p.set_defaults(func=_cmd_list)
    p = sub.add_parser("config", help="show the frozen configuration of this process")
    p.set_defaults(func=_cmd_config)
    p = sub.add_parser("report", help="report the current process state")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_cmd_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except TrainingMusaAdaptorError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return _EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
