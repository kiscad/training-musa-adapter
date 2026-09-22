"""Device-layer hooks: install the ``torch.cuda`` compat layer per framework.

Each framework installs the same idempotent helper at its own verified import
boundary, and a transformers-only run is a first-class scenario. Every hook
below calls :func:`backends.torch_cuda.ensure_cuda_compat`; whichever hook
fires first owns the undo, later hooks decline (``False`` -> skipped).

Triggers need a real exec body before the behavior the hook prepares;
namespace roots are not hook points. Verified against
megatron-core 0.16.1 and transformers 5.16.1 by tracing ``find_spec`` calls
alongside watcher installation:

- ``megatron`` is a *namespace* package in both the megatron-core wheel and
  the Megatron-LM source checkout (no ``__init__.py``): it cannot be a hook
  boundary; choose a concrete executable submodule.
- Order ``import torch; import megatron.core``: the watcher is installed at
  the end of ``import torch``, so ``megatron.core`` is the earliest legal
  megatron boundary.
- Order ``import megatron.core`` (no prior torch): ``megatron.core``'s own
  boundary is gone before torch finishes importing (torch is first imported
  inside ``megatron/core/tensor_parallel/cross_entropy.py`` line 5). The
  first boundary that still lands in time is ``megatron.core.parallel_state``
  (imported by cross_entropy.py line 7, and unconditionally by
  ``megatron/core/__init__.py``). All torch.cuda uses in the modules that
  execute earlier are runtime calls, not import-time, so the layer is still
  installed before any CUDA API is exercised.
- transformers: the root package is lazy and does not import torch, so the
  universal model boundary is ``transformers.modeling_utils`` (every model
  module imports ``PreTrainedModel`` from it -- verified first-catchable
  boundary even when the modeling module's own import triggered torch).
  The ``transformers`` root hook covers the dominant ``import torch;
  import transformers`` order at the earliest point.

Known gaps (recorded in docs/PATCH_LEDGER.md, not silently worked
around):

- In the megatron-first order the ``megatron.core`` hook never fires and
  stays reported ``pending``; the parallel_state hook does the work.
- If the very first torch import is triggered by a transformers *modeling*
  module (``import transformers.models....modeling_...`` before any
  ``import torch``/``import transformers``), the modeling module's own
  boundary is missed: the device layer still lands via
  ``transformers.modeling_utils``, but AttrPatches targeting that modeling
  module (e.g. the RMSNorm patch) cannot apply in that order.
- megatron source-checkout entry points that never import ``megatron.core``
  (there are none in 0.16.1 -- ``megatron.training`` imports torch first,
  then core) would need their own audited boundary.

Undo restores this project's overrides only: imported torch_musa/torchada
side effects remain (see backends/torch_cuda.py's mutation boundary).
"""

from __future__ import annotations

from .._engine import HookPatch
from ..backends import torch_cuda

__all__ = ["PATCHES"]

_STRATEGY = (
    "Delegate general CUDA-to-MUSA adaptation to torchada, then apply "
    "identity-tracked availability, tensor type, graph capture surface "
    "(graph class, capture context, pool handle, capture-state query), "
    "graph-safe RNG method restoration on the adapter's Generator "
    "proxy, and subclass transfer overrides; undo only project-owned "
    "bindings, not external adapter side effects or another caller's "
    "active layer. Declines (hook skipped) when the layer is already "
    "active or when no MUSA device is visible, so CPU/CUDA processes "
    "keep their original behavior and never import torchada."
)

_RATIONALE = (
    "Training frameworks require working CUDA APIs on MUSA, a truthful "
    "availability probe, CUDA-spelled Tensor.type() queries, graph classes "
    "and a routed capture context, graph-safe Generator methods their RNG "
    "trackers assert on the class, and device transfers that preserve "
    "TransformerEngine tensor subclasses."
)

_REMOVE_WHEN = (
    "The supported torchada/torch_musa stack provides CUDA adaptation plus "
    "MUSA availability, CUDA tensor type names, the routed graph capture "
    "surface (graph class, capture context, pool handle, capture-state "
    "query), graph-safe RNG methods visible on torch.Generator, and "
    "subclass-safe transfers with transfer options intact; verify the "
    "contracts in tests/test_torch_cuda.py without these overrides."
)

PATCHES = (
    HookPatch(
        id="torch.cuda.compat-layer",
        trigger="megatron.core",
        run=torch_cuda.ensure_cuda_compat,
        undo=torch_cuda.unapply,
        strategy=_STRATEGY
        + " Boundary: earliest legal megatron boundary in the dominant "
        "'import torch, then megatron' order (megatron itself is a "
        "namespace package, not an executable boundary).",
        rationale=_RATIONALE,
        upstream="torchada; torch_musa/core/tensor_attrs.py; Megatron CUDA consumers",
        remove_when=_REMOVE_WHEN,
    ),
    HookPatch(
        id="torch.cuda.compat-layer.megatron-late-boundary",
        trigger="megatron.core.parallel_state",
        run=torch_cuda.ensure_cuda_compat,
        undo=torch_cuda.unapply,
        strategy=_STRATEGY + " Boundary: first megatron boundary still reachable when "
        "'import megatron.core' itself triggers the first torch import "
        "(megatron.core and megatron.core.tensor_parallel boundaries are "
        "already gone); earlier modules make no import-time CUDA calls.",
        rationale=_RATIONALE,
        upstream="torchada; megatron/core/tensor_parallel/cross_entropy.py import chain",
        remove_when=_REMOVE_WHEN,
    ),
    HookPatch(
        id="torch.cuda.compat-layer.transformers",
        trigger="transformers",
        run=torch_cuda.ensure_cuda_compat,
        undo=torch_cuda.unapply,
        strategy=_STRATEGY
        + " Boundary: earliest legal transformers boundary in the dominant "
        "'import torch, then transformers' order; transformers-only runs "
        "install the device layer independently of megatron.",
        rationale=_RATIONALE,
        upstream="torchada; transformers 5.x lazy root package",
        remove_when=_REMOVE_WHEN,
    ),
    HookPatch(
        id="torch.cuda.compat-layer.transformers-late-boundary",
        trigger="transformers.modeling_utils",
        run=torch_cuda.ensure_cuda_compat,
        undo=torch_cuda.unapply,
        strategy=_STRATEGY
        + " Boundary: universal model boundary (every model module imports "
        "PreTrainedModel from transformers.modeling_utils); still reachable "
        "when 'import transformers' ran before torch, and even when the "
        "modeling module's own import triggered the first torch import.",
        rationale=_RATIONALE,
        upstream="torchada; transformers/models/* import chain",
        remove_when=_REMOVE_WHEN,
    ),
)
