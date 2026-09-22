"""mcore-bridge GDN patch: the same TileLang dispatch on the bridge binding.

mcore-bridge (the bridge ms-swift selects with --bridge_backend mcore-bridge)
subclasses Megatron's GatedDeltaNet with its own CP/SP-aware forward and
re-imports fla's ``chunk_gated_delta_rule`` into its own module namespace,
so patching Megatron's binding alone never reaches the execution path of a
Qwen3.5 mcore-bridge run.  mcore-bridge and Megatron-Bridge are different
projects; do not mix the names.

The dispatcher is shared with the Megatron patch via ``ops/gated_delta_rule.py``.
"""

from __future__ import annotations

from typing import Any

from ..._engine import AttrPatch
from ...ops.gated_delta_rule import make_tilelang_dispatcher

__all__ = ["PATCHES"]

_BRIDGE_GDN = "mcore_bridge.model.modules.gated_delta_net"


def _replace(original: Any) -> Any:
    return make_tilelang_dispatcher(original)


PATCHES = (
    AttrPatch(
        id="mcore_bridge.ssm.gated-delta-rule.tilelang",
        target=f"{_BRIDGE_GDN}:chunk_gated_delta_rule",
        rebind_prefixes=("mcore_bridge",),
        replace=_replace,
        rationale=(
            "mcore-bridge subclasses Megatron's GatedDeltaNet with its own "
            "CP/SP-aware forward and re-imports fla's chunk_gated_delta_rule "
            "into its own module namespace, so patching Megatron's binding "
            "alone never reaches the execution path of a Qwen3.5 mcore-bridge run."
        ),
        strategy=(
            "Install the same dispatcher on the bridge module's binding. It is "
            "an independent patch: selecting only one of the two leaves the "
            "other caller on FLA, and neither inspects the other's state. "
            "Alias repair scans mcore_bridge instead of megatron. The TileLang "
            "stack's version binding and JIT-cache requirements are the ones "
            "recorded on megatron.ssm.gated-delta-rule.tilelang."
        ),
        upstream=(
            "modelscope/mcore-bridge mcore_bridge/model/modules/gated_delta_net.py"
            ":chunk_gated_delta_rule"
        ),
        remove_when=(
            "Remove together with megatron.ssm.gated-delta-rule.tilelang, after "
            "re-verifying that the mcore-bridge forward no longer binds fla's "
            "symbol (or that the MUSA fast path ships where the bridge looks)."
        ),
    ),
)
