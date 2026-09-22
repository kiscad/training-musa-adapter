"""Megatron (megatron-core) patch suite.

Every patch in this package declares a ``megatron-core`` version gate next
to any dependency-specific gate (e.g. ``transformer_engine``).  The ranges
are calibrated against the upstream NVIDIA/Megatron-LM release branches in
the reference checkout (``core_v0.4.0`` ... ``core_v0.19.0``; the validated
adaptation base is ``core_v0.16.1`` == ``megatron-core 0.16.1``):

- the lower bound is the first release whose target contract (the module,
  attribute, signature or config field the wrapper codes against) exists.
  The engine reports an unresolvable target as ``failed``, not ``skipped``,
  so the gate must guarantee resolvability inside the declared range;
- the upper bound is the first release where the target is removed, renamed
  or its call contract visibly changes -- verified per target against the
  core_v0.17/0.18/0.19 branches, NOT a blanket "validated line" cap.
  Targets whose contract survives are capped at ``<0.20``: core_v0.19.0 is
  the newest release line in the checkout, so the range means
  "source-verified through the newest line"; re-verify when core_v0.20.0
  lands.

Where an interface change is visible at a release boundary, the adaptation
is split into a separate variant patch (own id, own gate) instead of
widening one wrapper's signature -- each patch stays simple and reviewable:

- ``megatron.moe.permutation.unfused-musa`` (<0.17) vs
  ``...unfused-musa.core17`` (>=0.17): permute/unpermute gain
  ``tokens_per_expert``/``align_size``/``pad_offsets`` in core_v0.17.0;
- ``megatron.embeddings.fused-rope-thd.apex`` (<0.17) vs
  ``...apex.core17`` (>=0.17): the dispatcher passes ``interleaved=`` to the
  thd kernel from core_v0.17.0.

Patches whose target lives in ``transformer_engine.*`` (not Megatron code)
use the Megatron lines where their Megatron-side consumers exist (verified
through core_v0.19.0) as their Megatron envelope.
"""
