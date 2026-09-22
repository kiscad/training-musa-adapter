"""Suite-scoped ONLY/DISABLE selection (patch id -> suite registry).

One entry naming a suite expands to every patch id of that suite, so a
whole adaptation domain (e.g. the Megatron suite) can be toggled at once.
"""

from __future__ import annotations

import pytest

from training_musa_adaptor._config import ConfigManager
from training_musa_adaptor._errors import ConfigError
from training_musa_adaptor.patches import PATCH_SUITES, PATCHES


def _suite_ids(suite: str) -> set[str]:
    return {pid for pid, name in PATCH_SUITES.items() if name == suite}


def _freeze(monkeypatch, suites=True, **env):
    for key, value in env.items():
        monkeypatch.setenv(f"TRAINING_MUSA_ADAPTOR_{key}", value)
    manager = ConfigManager()
    return manager.freeze(
        registered_patch_ids={patch.id for patch in PATCHES},
        patch_suites=PATCH_SUITES if suites else None,
    )


def test_disable_suite_expands_to_every_suite_id(monkeypatch):
    config = _freeze(monkeypatch, DISABLE="megatron")
    expected = _suite_ids("megatron")
    assert set(config.disable) == expected
    assert config.disable_suites == ("megatron",)
    for patch in PATCHES:
        if patch.id in expected:
            assert not config.patch_enabled(patch.id, set())
        elif patch.id.startswith(("transformers.", "mcore_bridge.")):
            assert config.patch_enabled(patch.id, set())


def test_env_disable_suite_replaces_file_list(monkeypatch, tmp_path):
    toml = tmp_path / "training-musa-adaptor.toml"
    toml.write_text(
        '[patches]\ndisable = ["transformers.qwen3-vl.text-rms-norm.fused-torch"]\n'
    )
    monkeypatch.setenv("TRAINING_MUSA_ADAPTOR_CONFIG", str(toml))
    monkeypatch.setenv("TRAINING_MUSA_ADAPTOR_DISABLE", "megatron")

    config = _freeze(monkeypatch, suites=True)

    # Environment replaces file lists entirely; the suite expands on top.
    assert "transformers.qwen3-vl.text-rms-norm.fused-torch" not in config.disable
    assert set(config.disable) == _suite_ids("megatron")


def test_only_suite_whitelists_the_whole_suite(monkeypatch):
    config = _freeze(monkeypatch, ONLY="transformers")
    assert set(config.only) == _suite_ids("transformers")
    for patch in PATCHES:
        assert config.patch_enabled(patch.id, set()) == (patch.id in config.only)
    assert config.notes  # ONLY whitelist mode is reported


def test_only_whitelist_ignores_disable_suite(monkeypatch):
    keep = "megatron.te.attention.capability-dispatch"
    config = _freeze(monkeypatch, ONLY=keep, DISABLE="megatron")
    assert config.only == (keep,)
    assert config.disable_suites == ("megatron",)  # recorded, but ignored
    assert config.patch_enabled(keep, set())


def test_unknown_suite_entry_reports_registered_ids(monkeypatch):
    with pytest.raises(ConfigError, match="unknown patch id\\(s\\) in ONLY/DISABLE"):
        _freeze(monkeypatch, DISABLE="megatron_typo")


def test_suite_entries_without_registry_stay_unknown(monkeypatch):
    """External registrations without a suite registry get the plain
    unknown-id error: expansion never guesses."""
    with pytest.raises(ConfigError, match="unknown patch id\\(s\\) in ONLY/DISABLE"):
        _freeze(monkeypatch, suites=False, DISABLE="megatron")
