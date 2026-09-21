"""The patch ledger.

Every patch this package applies lives in one of the modules below as
declarative data.  ``MODULES`` order matters for patches touching the same
symbol (earlier factories build the inner wrappers); the device
compatibility layer goes first so anything reading ``torch.cuda`` while its
own replacement is being built already sees a working namespace.

Adding a patch means adding a record in the matching module -- you should
never need to copy an upstream file into this repository.  Modules imported
here must stay standard-library + this package's lightweight modules at top
level; torch, frameworks and optional accelerator libraries load inside
factories or safe initialization functions.
"""

from __future__ import annotations

__all__ = ["PATCHES", "MODULES"]

# Populated as domains migrate (docs/MIGRATION_LEDGER.md tracks status):
#   from . import platform, transformer_engine
#   from .megatron import attention, layer_norm, rope, ssm
#   from .transformers import rms_norm
MODULES: tuple = ()

PATCHES = tuple(patch for module in MODULES for patch in module.PATCHES)
