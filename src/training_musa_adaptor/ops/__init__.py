"""Shared implementation-selection helpers (plain functions).

Only operators with at least two real call sites live here (design doc §6).
Standard-library only at import time; implementations load lazily.
"""

from __future__ import annotations
