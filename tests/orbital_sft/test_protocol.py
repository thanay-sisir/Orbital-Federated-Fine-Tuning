import json

import pytest

from evaluator.orbital_sft.invariants import check_receipts
from swarmshield.orbital.checkpoint import atomic_json, read
from swarmshield.orbital.config import Network, load, project_path
from swarmshield.orbital.events import EventQueue
from swarmshield.orbital.network import LinkEngine


def settings(**changes):
    return Network(
        rate_bps=80,
        chunk_bytes=10,
        framing_bytes=0,
        ack_bytes=0,
        acquisition_seconds=0,
        queue_bytes=1000,
        byte_budget=1000,
        **changes,
    )


def contacts(client=0, station="ground", start=0, end=10):
    return {"client": client, "station": station, "start": start, "end": end, "delay_seconds": 0}


@pytest.mark.parametrize(
    "name",
    [
        "compatibility",
        "two_client_smoke",
        "five_gpu",
        "one_gpu_reference",
        "two_gpu",
        "connected",
        "contact_aware",
    ],
)
def test_configuration_templates(name):
    cfg = load(f"configs/orbital_sft/{name}.yaml")
    cfg.ready()
    assert cfg.training.workers <= cfg.training.clients
    assert cfg.model.target_modules


def test_escape_rejected():
    with pytest.raises(ValueError):
        project_path("../outside")


def test_chunks_resume_across_contacts():
    engine = LinkEngine([contacts(end=1), contacts(start=5, end=7)], settings())
    receipt = engine.transfer(0, 30, 0, 7, "upload", "a")
    assert receipt["complete"] and receipt["arrival"] == 7
    assert [c["offset"] for c in receipt["chunks"]] == [0, 10, 20]
    assert check_receipts(engine.receipts, 80, 0) == 30


def test_unfinished_payload_never_delivered():
    engine = LinkEngine([contacts(end=1)], settings())
    result = engine.transfer(0, 20, 0, 5, "upload", "a")
    assert result["verified_bytes"] == 10
    assert not result["complete"] and result["arrival"] is None


def test_station_and_client_half_duplex():
    engine = LinkEngine([contacts(), contacts(client=1)], settings())
    first = engine.transfer(0, 20, 0, 10, "upload", "a")
    second = engine.transfer(1, 20, 0, 10, "download", "b")
    assert second["chunks"][0]["start"] == first["arrival"]
    check_receipts(engine.receipts, 80, 0)


def test_reservations_leave_earlier_gaps_usable():
    engine = LinkEngine([contacts(), contacts(client=1)], settings())
    engine.transfer(0, 10, 5, 10, "upload", "future")
    early = engine.transfer(1, 10, 0, 10, "upload", "early")
    assert early["arrival"] == 1
    check_receipts(engine.receipts, 80, 0)


def test_duplicate_transfer_rejected():
    engine = LinkEngine([contacts()], settings())
    engine.transfer(0, 10, 0, 10, "upload", "a")
    with pytest.raises(ValueError, match="Duplicate"):
        engine.transfer(0, 10, 0, 10, "upload", "a")


def test_round_boundary_link_resume():
    engine = LinkEngine([contacts()], settings())
    engine.transfer(0, 10, 0, 10, "upload", "a")
    restored = LinkEngine([contacts()], settings())
    restored.restore(json.loads(json.dumps(engine.snapshot())))
    lhs = engine.transfer(0, 10, 0, 10, "upload", "b")
    rhs = restored.transfer(0, 10, 0, 10, "upload", "b")
    assert lhs == rhs


def test_stable_events_and_causality():
    queue = EventQueue()
    queue.push(3, "close", {}, 10)
    queue.push(3, "delivery", {}, 1)
    queue.push(3, "second_delivery", {}, 1)
    assert [e.kind for e in queue.drain()] == ["delivery", "second_delivery", "close"]
    with pytest.raises(ValueError):
        queue.push(2, "past", {})


def test_budget_and_framing():
    cfg = Network(
        rate_bps=80,
        chunk_bytes=10,
        framing_bytes=2,
        ack_bytes=2,
        acquisition_seconds=0,
        byte_budget=20,
    )
    engine = LinkEngine([contacts()], cfg)
    result = engine.transfer(0, 20, 0, 10, "upload", "a")
    assert not result["complete"] and engine.wire_bytes == 14
    check_receipts(engine.receipts, 80, 0)


def test_checkpoint_atomic(tmp_path):
    path = tmp_path / "checkpoint.json"
    atomic_json(path, {"version": 2, "event_queue": []})
    assert read(path)["version"] == 2
    assert not path.with_suffix(".json.tmp").exists()
