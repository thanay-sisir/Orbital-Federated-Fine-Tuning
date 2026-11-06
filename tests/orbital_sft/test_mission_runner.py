import hashlib
from pathlib import Path

import pytest

from swarmshield.orbital.checkpoint import read
from swarmshield.orbital.config import Network, load, digest
from swarmshield.orbital.feasibility import simulate
from swarmshield.orbital.mission_runner import apply_progress, bind_receipt
from swarmshield.orbital.network import LinkEngine
from swarmshield.orbital.round_planner import plan_round
from swarmshield.posttraining.trainer import local_steps


@pytest.mark.parametrize("policy", ["connected", "fixed", "contact_aware", "feasible_fair"])
def test_coupled_planner_reproduces_archived_case62(policy):
    root = Path(__file__).resolve().parents[2]
    contact_file = root / "results/orbital/reference-22367142/contacts.json"
    if not contact_file.exists():
        pytest.skip("Archived geometry fixture not staged")
    cfg = load("configs/orbital_sft/five_gpu_one_epoch.yaml")
    contacts = [c for c in read(contact_file)["contacts"] if c["station"] == cfg.orbit.stations[0].name]
    if policy == "connected":
        contacts = [{"client": i, "station": "connected-ground", "start": 0, "end": 86400, "delay_seconds": 0} for i in range(5)]
    network = cfg.network.model_copy(update={"rate_bps": 1000000, "min_participants": 2})
    size = (root / "results/orbital/reference-22367142/round-0000/global_B.safetensors").stat().st_size
    engine = LinkEngine(contacts, network)
    age, count, steps = [0] * 5, 0, 0
    for round_id in range(6):
        engine, tasks, summary = plan_round(engine, 5, round_id * 14400, (round_id + 1) * 14400, size, 60, 4, policy, age, round_id)
        accepted = set(summary["accepted_clients"])
        count += len(accepted)
        steps += summary["accepted_steps"]
        for client in range(5):
            age[client] = 0 if client in accepted else age[client] + 1
        assert len({t["client"] for t in tasks}) == len(tasks)
    expected = read(root / f"results/orbital/mission-feasibility-v1/{policy}/case-062.json")
    assert count == expected["accepted_updates"]
    assert steps == expected["accepted_steps"]
    assert engine.wire_bytes == expected["wire_bytes"]


def test_planned_steps_cannot_exceed_training_budget():
    cfg = load("configs/orbital_sft/five_gpu.yaml")
    assert local_steps(cfg, {"local_steps": 2}) == 2
    for invalid in (0, 5, True, 2.0):
        with pytest.raises(ValueError):
            local_steps(cfg, {"local_steps": invalid})


def test_receipts_bind_real_bytes_and_reject_a_size_change(tmp_path):
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"abcdefgh")
    settings = Network(rate_bps=800, chunk_bytes=4, framing_bytes=0, ack_bytes=0, acquisition_seconds=0)
    engine = LinkEngine([{"client": 0, "station": "s", "start": 0, "end": 10, "delay_seconds": 0}], settings)
    receipt = engine.transfer(0, 8, 0, 10, "upload", "test")
    bind_receipt(receipt, payload)
    assert receipt["sha256"] == digest(payload)
    assert receipt["chunks"][0]["sha256"] == hashlib.sha256(b"abcd").hexdigest()
    payload.write_bytes(b"short")
    with pytest.raises(ValueError, match="byte budget"):
        bind_receipt(receipt, payload)


def test_failed_aggregation_advances_performed_work_but_not_accepted_work():
    state = {"cursors": [0, 0], "age": [0, 0], "performed_examples": 0, "performed_steps": 0, "delivered_steps": 0, "accepted_steps": 0, "accepted_examples": 0, "accepted_updates": [0, 0], "accepted_digests": [[], []]}
    result = {"client": 0, "cursor": 8, "steps": 2, "receipt": {"complete": True}, "example_digests": ["a", "b"]}
    apply_progress(state, [result], [])
    assert state["cursors"] == [8, 0]
    assert state["performed_examples"] == 8
    assert state["performed_steps"] == state["delivered_steps"] == 2
    assert state["accepted_steps"] == state["accepted_examples"] == 0
    assert state["accepted_digests"] == [[], []]


def test_fake_training_round_runs_with_actual_aggregation(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import load_file, save_file
    import swarmshield.orbital.mission_runner as runner
    import swarmshield.posttraining.trainer as trainer

    cfg = load("configs/orbital_sft/five_gpu.yaml")
    cfg.training.clients = cfg.training.workers = 2
    cfg.training.rounds = 1
    cfg.training.evaluate_test = False
    cfg.network.policy = "connected"
    cfg.network.round_seconds = cfg.orbit.mission_seconds = 1000
    calls = []

    def initialize(config, device, output):
        path = Path(output)
        path.mkdir()
        save_file({"layer.lora_A": torch.ones(2), "layer.lora_B": torch.zeros(2)}, str(path / "adapter.safetensors"))
        return str(path / "adapter.safetensors")

    def execute(task):
        calls.append(task)
        path = Path(task["output"])
        path.mkdir()
        state = load_file(task["global_adapter"])
        state["layer.lora_B"] += task["local_steps"]
        save_file({"layer.lora_B": state["layer.lora_B"]}, str(path / "upload.safetensors"))
        from swarmshield.posttraining.model_loader import fixed_a_hash
        return {"client": task["client"], "version": task["version"], "base_identity": cfg.model.identity + "@" + cfg.model.revision, "unique_examples": 16, "example_digests": [f"c{task['client']}"], "steps": task["local_steps"], "cursor": task["cursor"] + task["local_steps"] * cfg.training.accumulation, "upload_path": str(path / "upload.safetensors"), "upload_sha256": digest(path / "upload.safetensors"), "fixed_A_sha256": fixed_a_hash(state), "adapter_contract": cfg.model.model_dump()}

    class Immediate:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class Worker:
        def submit(self, fn, *args):
            return Immediate(fn(*args))

        def shutdown(self, **kwargs):
            pass

    run_path = tmp_path / "run"

    def make_run(*args):
        run_path.mkdir()
        return run_path

    contacts = tmp_path / "contacts.json"
    contacts.write_text('{"contacts": []}')
    monkeypatch.setattr(runner, "require_allocation", lambda cfg: None)
    monkeypatch.setattr(runner, "compatibility", lambda cfg: None)
    monkeypatch.setattr(runner, "make_run", make_run)
    monkeypatch.setattr(runner, "pool", Worker)
    monkeypatch.setattr(runner, "generate", lambda cfg: contacts)
    monkeypatch.setattr(runner, "source_identity", lambda: {})
    monkeypatch.setattr(runner, "project_path", lambda path: Path(path))
    monkeypatch.setattr(trainer, "initialize", initialize)
    monkeypatch.setattr(trainer, "execute", execute)
    monkeypatch.setattr(trainer, "assess", lambda *args: {"validation": {"answer_loss": 1.0}})
    runner.run_mission(cfg, "test")
    assert len(calls) == 2
    accounting = read(run_path / "work_accounting.json")
    assert accounting["accepted_steps"] == 8
    assert accounting["performed_examples"] == accounting["accepted_examples"] == 32
    assert read(run_path / "health.json")["status"] == "completed"
    assert torch.equal(load_file(accounting["global_adapter"])["layer.lora_B"], torch.full((2,), 4.0))
