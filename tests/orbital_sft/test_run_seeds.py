from pathlib import Path

from swarmshield.orbital.config import load, project_path
from swarmshield.orbital.checkpoint import read


def test_training_seed_does_not_change_the_frozen_data_contract():
    cfg = load("configs/orbital_sft/mission_case62/fixed.yaml")
    original_data = cfg.data.model_dump()
    assert cfg.seed == cfg.training_seed == 42
    cfg.training.run_seed = 43
    assert cfg.seed == 42
    assert cfg.training_seed == 43
    assert cfg.data.model_dump() == original_data
    prepared = read(Path(__file__).resolve().parents[1] / "fixtures/prepared_manifest.json")
    assert prepared["seed"] == cfg.seed
    assert prepared["data"] == cfg.data.model_dump()
    assert prepared["model"] == cfg.model.model_dump()
    values = cfg.model_dump()
    assert values["seed"] == 42 and values["training"]["run_seed"] == 43


def test_zero_seed_is_valid_and_does_not_fall_back_to_data_seed():
    cfg = load("configs/orbital_sft/mission_case62/fixed.yaml")
    cfg.training.run_seed = 0
    assert cfg.training_seed == 0


def test_seed_specific_initialization_is_reproducible_on_cpu():
    import torch

    cfg = load("configs/orbital_sft/mission_case62/fixed.yaml")
    draws = []
    for seed in (42, 43, 42):
        cfg.training.run_seed = seed
        torch.manual_seed(cfg.training_seed)
        draws.append(torch.randn(32))
    assert torch.equal(draws[0], draws[2])
    assert not torch.equal(draws[0], draws[1])
