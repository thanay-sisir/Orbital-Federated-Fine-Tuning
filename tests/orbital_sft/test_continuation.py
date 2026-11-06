import json

import pytest

from swarmshield.orbital.config import load, digest
from swarmshield.orbital.continuation import completed_adapter
from swarmshield.posttraining.benchmarks import numeric_answer


def test_gsm_scoring_requires_a_final_answer_marker():
    assert numeric_answer("I counted 17 things") is None
    assert numeric_answer("work 99\n#### 1,234.00") == numeric_answer("#### 1234")
    assert numeric_answer("#### -3.5") == "-3.5"


def test_parent_must_be_complete_and_unchanged(tmp_path, monkeypatch):
    import swarmshield.orbital.continuation as continuation

    monkeypatch.setattr(continuation, "project_path", lambda value: tmp_path / value)
    cfg = load("configs/orbital_sft/five_gpu_one_epoch.yaml")
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "manifest.json").write_text(json.dumps({"config": cfg.model_dump()}))
    (parent / "health.json").write_text('{"status":"running"}')
    with pytest.raises(ValueError, match="has not completed"):
        completed_adapter("parent", cfg, 1)
    (parent / "health.json").write_text('{"status":"completed"}')
    (parent / "coverage.json").write_text(json.dumps({"all_clients_aggregated": True, "processed_examples": [3], "shard_examples": [3]}))
    adapter = parent / "adapter.safetensors"
    adapter.write_bytes(b"unchanged")
    (parent / "checkpoint.json").write_text(json.dumps({"global_adapter": "parent/adapter.safetensors", "global_sha256": digest(adapter)}))
    assert completed_adapter("parent", cfg, 1)[0] == adapter
    adapter.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        completed_adapter("parent", cfg, 1)
    with pytest.raises(ValueError, match="epoch number"):
        completed_adapter("parent", cfg, 2)


def test_continuation_initializes_from_parent_weights(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import load_file, save_file
    import swarmshield.posttraining.trainer as trainer

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
            self.lora_B = torch.nn.Parameter(torch.zeros(2))

        def save_pretrained(self, *args, **kwargs):
            pass

    model = Adapter()
    expected = {"lora_A": torch.tensor([1.0, 2.0]), "lora_B": torch.tensor([3.0, 4.0])}
    save_file(expected, str(tmp_path / "parent.safetensors"))
    monkeypatch.setattr(trainer, "project_path", lambda value: tmp_path / value)
    monkeypatch.setattr(trainer, "load_model", lambda cfg, device: model)
    monkeypatch.setattr(trainer, "_MODEL", None)
    monkeypatch.setattr(trainer, "_CONFIG", None)
    cfg = load("configs/orbital_sft/epoch2.yaml")
    result = trainer.initialize(cfg.model_dump(), "cpu", "initialized", "parent.safetensors")
    actual = load_file(result)
    assert actual.keys() == expected.keys()
    assert all(torch.equal(actual[key], expected[key]) for key in expected)
