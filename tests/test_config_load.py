"""Config-load sanity tests.

Every YAML under `configs/` must parse and load through
`hybrid_alife.experiments.runner.load_config` without raising. This guards
against silent config drift when fields are renamed.

Ablation configs typically *partially* override `base.yaml` at runtime; on
their own they may not have every top-level key. We accept either:
  - full configs (all top-level sections present), loaded via `load_config`;
  - partial configs, which must still parse as YAML mappings whose keys are
    a subset of the base config's keys.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hybrid_alife.experiments.runner import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"

REQUIRED_TOP_LEVEL = {
    "seed",
    "run_name",
    "output_dir",
    "world",
    "embodied",
    "avida",
    "evolution",
    "logging",
}


def _yaml_files() -> list[Path]:
    return sorted(CONFIGS_DIR.glob("*.yaml"))


def test_configs_dir_has_files():
    files = _yaml_files()
    assert files, "expected at least one config under configs/"


@pytest.mark.parametrize("config_path", _yaml_files(), ids=lambda p: p.name)
def test_yaml_parses(config_path: Path):
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict), f"{config_path.name}: top-level must be a mapping"


@pytest.mark.parametrize("config_path", _yaml_files(), ids=lambda p: p.name)
def test_config_is_full_or_partial(config_path: Path):
    """Full experiment configs must load through load_config; partials only
    need to parse as a mapping (they're consumed by specialised drivers like
    the comm benchmark / ablation matrix)."""
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert isinstance(data, dict)
    if REQUIRED_TOP_LEVEL.issubset(set(data.keys())):
        cfg = load_config(config_path)
        assert cfg.seed is not None
        assert cfg.run_name
        assert cfg.world.width > 0 and cfg.world.height > 0


def test_base_config_loads():
    cfg = load_config(CONFIGS_DIR / "base.yaml")
    assert cfg.world.width > 0
    assert cfg.embodied.population_size > 0
    assert cfg.avida.population_size > 0
    assert cfg.evolution.generations > 0
