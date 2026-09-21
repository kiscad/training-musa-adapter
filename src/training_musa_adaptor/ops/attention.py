"""Attention implementation selection (plain functions, design doc §6).

The candidate implementations are operator-local names (NOT global provider
IDs).  Only entries with a verified integration and tests join the usable
list; the rest stay declared-but-declined so the capability matrix reflects
reality.

This module is standard-library only at import time.  The actual
implementations land with the megatron attention patch migration (S2).
"""

from __future__ import annotations

__all__ = ["DECLARED_IMPLEMENTATIONS", "DEFAULT_CANDIDATE_ORDER"]

#: All implementation names this call site knows about (config validation).
DECLARED_IMPLEMENTATIONS = (
    "mudnn",
    "mate",
    "flash_attn",
    "te_unfused",
    "torch_sdpa_math",
)

#: Fixed candidate order for the verified stack (torch 2.7.1 / torch_musa
#: 2.7.1 / MT-TE 2.0.0, muDNN v3107); it is evidence-backed for this stack,
#: not a universal MUSA performance ranking (design doc §6.2/§6.4).
DEFAULT_CANDIDATE_ORDER = ("mudnn", "mate", "te_unfused")
