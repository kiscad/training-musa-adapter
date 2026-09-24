"""Non-throwing torchao availability probe for peft's LoRA dispatch on MUSA.

peft 0.19.0 raised ``peft.import_utils.is_torchao_available``'s minimum
torchao from 0.4.0 to 0.16.0, and made the probe *raise* ``ImportError``
when the installed torchao is older.  The probe is not only used as an
opt-in check: ``peft/tuners/lora/torchao.py:dispatch_torchao`` sits in
``LoraModel._create_new_module``'s dispatcher chain and calls it for every
adapter target before the torchao weight-type check can decline, so with a
torchao that is present but older than 0.16.0, *every* ``get_peft_model``
call crashes with ``ImportError`` -- not just torchao-quantized ones.

The MUSA training stack ships the community torchao build 0.9.0 (a real
torchao, compiled for this stack, but older than peft's new minimum), so
ms-swift LoRA SFT fails at adapter creation before any training step.  The
probe's own contract one version earlier (peft 0.14.0-0.18.x, minimum
0.4.0) accepted this exact torchao and returned True/False as a plain
predicate; this patch restores that predicate behavior for the too-old
case instead of weakening peft's quantization requirements anywhere else:

- torchao missing -> False (unchanged);
- torchao present and >= peft's minimum -> True (unchanged, original path);
- torchao present but older than peft's minimum -> False, so the
  dispatcher falls through to ``dispatch_default`` and LoRA works;
  torchao-specific LoRA paths stay unavailable, which is truthful --
  peft 0.19+'s torchao adapter API genuinely needs torchao >= 0.16.

The patch is deliberately narrow: it does not touch the minimum version,
does not fake availability, and leaves every non-torchao ``ImportError``
to propagate.

Boundary note: ``peft.tuners.lora.torchao`` binds the probe with
``from peft.import_utils import is_torchao_available`` at its own import
time.  The AttrPatch applies at the ``peft.import_utils`` execution
boundary, which is always before ``peft.tuners.lora.torchao`` imports in
peft's own import graph, so the dispatcher sees the wrapped probe.  A
process that already imported ``peft.tuners.lora.torchao`` before this
adaptor activates keeps the raising alias (documented late-apply caveat).

Version gates: the raising probe exists from peft 0.14.0, but its minimum
is 0.4.0 through 0.18.x -- torchao 0.9.0 passes there and no crash is
possible, so the patch is scoped to the range that can fail.  Source
check (sdists): 0.19.0/0.19.1/0.20.0/0.21.0 all raise with the same
message and minimum 0.16.0.  0.22 does not exist yet; the upper bound is
a conservative verification boundary, not an observed incompatibility.
"""

from __future__ import annotations

import functools
from typing import Any

from .._engine import AttrPatch

__all__ = ["PATCHES"]


def replace_torchao_probe(original: Any) -> Any:
    """Report torchao as unavailable when peft would reject its version.

    The original raises ``ImportError`` for a present-but-too-old torchao;
    that raise inside the LoRA dispatcher chain makes every adapter target
    fail, so the too-old case degrades to ``False`` here.  Everything else
    -- missing torchao, sufficient versions, and any non-torchao import
    error -- is the original's answer.
    """

    @functools.wraps(original)
    def is_torchao_available() -> bool:
        try:
            return original()
        except ImportError as exc:
            # The only ImportError the probe raises on its own is the
            # present-but-too-old torchao case; its message names torchao.
            if "torchao" in str(exc):
                return False
            raise

    return is_torchao_available


PATCHES = (
    AttrPatch(
        id="peft.lora.torchao-probe.version-compat",
        target="peft.import_utils:is_torchao_available",
        replace=replace_torchao_probe,
        version_gates=("peft >=0.19,<0.22",),
        rationale=(
            "peft 0.19.0 raised is_torchao_available's minimum torchao from "
            "0.4.0 to 0.16.0 and the probe raises ImportError on older "
            "torchao; peft/tuners/lora/torchao.py:dispatch_torchao calls the "
            "probe for every LoRA target in _create_new_module's dispatcher "
            "chain, so the MUSA stack's community torchao 0.9.0 build turns "
            "every get_peft_model call (plain LoRA included) into an "
            "ImportError before training starts."
        ),
        strategy=(
            "Wrap the probe only: present-but-too-old torchao reports "
            "False so the dispatcher falls through to dispatch_default and "
            "plain LoRA works; missing torchao and sufficient versions keep "
            "the original answer, and the raise (with its message) is kept "
            "for any non-torchao import error. No minimum version is "
            "changed and torchao-specific LoRA paths stay unavailable, "
            "which is truthful for peft 0.19+'s torchao>=0.16 adapter API. "
            "torch is never imported. Boundary: the patch applies at the "
            "peft.import_utils exec boundary, before "
            "peft.tuners.lora.torchao binds the probe by-value; a process "
            "that imported the dispatcher earlier keeps the raising alias "
            "(late-apply caveat). "
            "TRAINING_MUSA_ADAPTOR_DISABLE=peft.lora.torchao-probe.version-compat "
            "restores the raising probe."
        ),
        upstream=(
            "huggingface/peft src/peft/import_utils.py:is_torchao_available "
            "(0.19.0-0.21.0); src/peft/tuners/lora/torchao.py:dispatch_torchao"
        ),
        remove_when=(
            "Remove when the MUSA stack ships torchao >= peft's minimum, or "
            "when peft turns the version probe back into a non-raising "
            "predicate (both verified by running a plain LoRA SFT step with "
            "this patch disabled and torchao installed)."
        ),
    ),
)
