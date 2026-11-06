import json
from dataclasses import fields

import pytest

from swarmshield.orbital.config import Network
from swarmshield.orbital.outages import OutageEngine, PolicyView, audit, plan_outage_round


def setup(events=None, clients=1, mission=20, budget=10000, quorum=1, stations=("s",)):
    contacts = [{"client": c, "station": s, "start": 0, "end": mission, "delay_seconds": 0} for c in range(clients) for s in stations]
    settings = Network(rate_bps=80, chunk_bytes=10, framing_bytes=0, ack_bytes=0,
                       acquisition_seconds=0, byte_budget=budget, min_participants=quorum)
    trace = {"schema_version": 1, "seed": 101, "profile": "fixture", "events": events or []}
    return OutageEngine(contacts, settings, trace, clients, mission)


def run(engine, deadline=10, policy="fixed", clients=1, step=1, steps=1):
    result = plan_outage_round(engine, clients, 0, deadline, 20, step, steps, policy, [0] * clients, 0)
    audit(engine, deadline)
    return result[1:]


@pytest.mark.parametrize("policy", ["fixed", "retry_backoff", "feasible_fair"])
def test_no_outage_delivery_respects_capacity_and_deadline(policy):
    engine = setup(mission=100)
    tasks, summary = run(engine, deadline=100, policy=policy)
    assert summary["accepted_steps"] == 1
    assert tasks[0]["steps"] == 1
    assert engine.wire_bytes == 40
    assert engine.mission_clock == 100


def test_failed_attempt_consumes_bytes_and_can_miss_mission_deadline():
    normal = setup()
    assert run(normal, deadline=5)[1]["accepted_steps"] == 1
    failed = setup([{"kind": "link", "client": 0, "start": 0.5, "end": 2}])
    tasks, summary = run(failed, deadline=5)
    assert summary["accepted_steps"] == 0
    assert summary["failed_attempts"] == 1
    assert failed.attempts[0]["wire_bytes"] == 5
    assert failed.wire_bytes == 25
    assert failed.mission_clock == 5


def test_verified_prefix_is_retained_and_failed_chunk_is_retried():
    engine = setup([{"kind": "link", "client": 0, "start": 1.5, "end": 3}])
    tasks, summary = run(engine)
    down = next(r for r in engine.receipts if r["direction"] == "download")
    assert down["complete"] and [r["offset"] for r in down["chunks"]] == [0, 10]
    assert down["chunks"][0]["start"] == 0
    assert down["chunks"][1]["start"] == 3
    assert summary["accepted_steps"] == 1 and engine.wire_bytes == 45


def test_boundary_outage_is_observed_before_any_decision_or_compute():
    engine = setup([{"kind": "station", "station": "s", "start": 0, "end": 3}])
    run(engine)
    assert engine.decisions[0]["time"] == 3
    assert engine.decisions[0]["observations_seen"] == 2
    assert engine.attempts[0]["start"] == 3
    assert not any(not a["success"] for a in engine.attempts)


def test_client_downtime_restarts_unfinished_compute_and_spends_mission_time():
    engine = setup([{"kind": "client", "client": 0, "start": 2.5, "end": 4}])
    tasks, summary = run(engine, deadline=7, step=2)
    assert tasks[0]["complete"] == 6
    assert summary["wasted_compute_seconds"] == 0.5
    assert summary["accepted_steps"] == 0
    assert summary["scheduled_steps"] == 1


def test_overlapping_outages_do_not_restore_a_client_early():
    engine = setup([{"kind": "link", "client": 0, "start": 0.5, "end": 3},
                    {"kind": "link", "client": 0, "start": 2, "end": 4}])
    run(engine)
    assert engine.attempts[1]["start"] == 4


def test_policy_has_no_realized_future_outage_input(monkeypatch):
    import swarmshield.orbital.outages as module

    seen = []
    original = module.admit_steps

    def inspect(view, policy):
        assert {f.name for f in fields(view)} == {f.name for f in fields(PolicyView)}
        assert not any("trace" in f.name or "outage" in f.name for f in fields(view))
        seen.append(view)
        return original(view, policy)

    monkeypatch.setattr(module, "admit_steps", inspect)
    plain = setup(mission=100)
    future = setup([{"kind": "station", "station": "s", "start": 0.5, "end": 2}], mission=100)
    run(plain, deadline=100, policy="feasible_fair")
    run(future, deadline=100, policy="feasible_fair")
    assert seen[0] == seen[1]
    assert plain.decisions == future.decisions
    assert future.wire_bytes > plain.wire_bytes


def test_budget_counts_failed_bytes_and_no_partial_update_is_accepted():
    engine = setup([{"kind": "link", "client": 0, "start": 0.5, "end": 2}], budget=40)
    tasks, summary = run(engine)
    assert summary["accepted_steps"] == 0
    assert engine.wire_bytes <= 40
    assert engine.wire_bytes == sum(a["wire_bytes"] for a in engine.attempts)


def test_quorum_failure_does_not_turn_delivered_work_into_accepted_work():
    engine = setup(clients=2, quorum=2)
    engine.contacts = [c for c in engine.contacts if c["client"] == 0]
    tasks, summary = run(engine, clients=2)
    assert summary["delivered_clients"] == [0]
    assert summary["accepted_clients"] == [] and summary["accepted_steps"] == 0


def test_shared_station_and_two_station_half_duplex():
    engine = setup([{"kind": "station", "station": "s", "start": 0.5, "end": 2}], clients=2, stations=("s", "t"), quorum=2)
    run(engine, clients=2)
    assert audit(engine, 10)["failed_attempts"] >= 1


def test_round_snapshot_restores_clock_and_rejects_a_different_trace():
    engine = setup([{"kind": "client", "client": 0, "start": 4, "end": 7}])
    run(engine)
    restored = setup([{"kind": "client", "client": 0, "start": 4, "end": 7}])
    restored.restore(json.loads(json.dumps(engine.snapshot())))
    lhs = plan_outage_round(engine, 1, 10, 20, 20, 1, 1, "fixed", [0], 1)[2]
    rhs = plan_outage_round(restored, 1, 10, 20, 20, 1, 1, "fixed", [0], 1)[2]
    assert lhs == rhs and engine.snapshot() == restored.snapshot()
    with pytest.raises(ValueError, match="trace changed"):
        setup().restore(restored.snapshot())
    with pytest.raises(ValueError, match="round"):
        plan_outage_round(restored, 1, 0, 5, 20, 1, 1, "fixed", [0], 2)


def test_actual_coordinator_matches_outage_work_plan(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import load_file, save_file
    import swarmshield.orbital.resilient_mission as runner
    import swarmshield.posttraining.trainer as trainer
    from swarmshield.orbital.checkpoint import atomic_json, read
    from swarmshield.orbital.config import load
    from swarmshield.posttraining.model_loader import fixed_a_hash

    cfg = load("configs/orbital_sft/recovery_v1.yaml")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    atomic_json(prepared / "manifest.json", {"fixture": "synthetic CPU outage protocol test"})
    cfg.data.prepared_path = str(prepared.relative_to(runner.project_path(".")))
    cfg.training.clients = cfg.training.workers = 2
    cfg.training.rounds = 3
    cfg.training.local_steps = 2
    cfg.training.accumulation = 2
    cfg.network.policy = "fixed"
    cfg.network.round_seconds = 1000
    cfg.orbit.mission_seconds = 3000
    run_path = tmp_path / "run"
    geometry = tmp_path / "contacts.json"
    atomic_json(geometry, {"contacts": [{"client": c, "station": "s", "start": 0, "end": 3000, "delay_seconds": 0} for c in range(2)]})
    trace = tmp_path / "trace.json"
    atomic_json(trace, {"schema_version": 1, "seed": 1, "profile": "fixture", "events": [{"kind": "station", "station": "s", "start": 0, "end": 2000}]})

    def make_run(*args):
        run_path.mkdir()
        atomic_json(run_path / "manifest.json", {})
        return run_path

    def initialize(config, device, output, parent=None):
        from pathlib import Path
        path = Path(output)
        path.mkdir()
        save_file(load_file(parent) if parent else {"layer.lora_A": torch.ones(2), "layer.lora_B": torch.zeros(2)}, str(path / "adapter.safetensors"))
        return str(path / "adapter.safetensors")

    def execute(task):
        from pathlib import Path
        path = Path(task["output"])
        path.mkdir()
        state = load_file(task["global_adapter"])
        weights = torch.nn.Parameter(state["layer.lora_B"].clone())
        optimizer = torch.optim.SGD([weights], lr=0.1)
        for _ in range(task["local_steps"]):
            optimizer.zero_grad()
            ((weights - 1) ** 2).sum().backward()
            optimizer.step()
        save_file({"layer.lora_B": weights.detach()}, str(path / "upload.safetensors"))
        count = task["local_steps"] * cfg.training.accumulation
        return {"client": task["client"], "version": task["version"], "base_identity": cfg.model.identity + "@" + cfg.model.revision,
                "steps": task["local_steps"], "cursor": task["cursor"] + count, "unique_examples": count, "example_digests": [str(task["client"])],
                "upload_path": str(path / "upload.safetensors"), "upload_sha256": runner.digest(path / "upload.safetensors"),
                "fixed_A_sha256": fixed_a_hash(state), "adapter_contract": cfg.model.model_dump()}

    class Worker:
        def submit(self, function, *args):
            from concurrent.futures import Future
            future = Future()
            future.set_result(function(*args))
            return future

        def shutdown(self, **kwargs):
            pass

    monkeypatch.setattr(runner, "make_run", make_run)
    monkeypatch.setattr(runner, "require_allocation", lambda cfg: None)
    monkeypatch.setattr(runner, "compatibility", lambda cfg: None)
    monkeypatch.setattr(runner, "source_identity", lambda: {})
    monkeypatch.setattr(runner, "generate", lambda cfg: geometry)
    monkeypatch.setattr(runner, "pool", Worker)
    monkeypatch.setattr(trainer, "initialize", initialize)
    monkeypatch.setattr(trainer, "execute", execute)
    monkeypatch.setattr(trainer, "assess", lambda *args: {"validation": {"answer_loss": 1.0}})
    runner.run_resilient_mission(cfg, "fixture", outage_trace=str(trace))
    state = read(run_path / "work_accounting.json")
    assert state["version"] == 1 and state["accepted_steps"] == state["performed_steps"] == 4
    assert state["cursors"] == [4, 4]
    assert read(run_path / "outage_execution.json")["mission_clock"] == 3000
