"""The shipped config sample must stay valid against the live config schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from training_musa_adaptor._config import ConfigManager
from training_musa_adaptor._errors import ConfigError

SAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "configs"
    / "training-musa-adaptor.toml"
)


def test_sample_parses_and_yields_stock_defaults():
    """The sample as shipped equals built-in defaults; TOML fields are accepted."""
    from training_musa_adaptor.patches import PATCHES

    manager = ConfigManager()
    manager.set_config_path(str(SAMPLE))
    config = manager.freeze(registered_patch_ids={patch.id for patch in PATCHES})

    attention = config.attention()
    assert (attention.policy, attention.implementations, attention.fallback) == (
        "auto",
        (),
        "reference",
    )
    assert config.only == () and config.disable == ()
    assert config.sources["attention.policy"] == "file"
    assert config.sources["patches.only"] == "defaults"


def test_sample_rejects_schema_drift_loudly(tmp_path):
    """A field not in the schema must error, so sample drift cannot ship."""
    broken = tmp_path / "broken.toml"
    broken.write_text("bogus_section = true\n\n" + SAMPLE.read_text())
    manager = ConfigManager()
    manager.set_config_path(str(broken))
    with pytest.raises(ConfigError, match="unknown top-level section"):
        manager.freeze()
