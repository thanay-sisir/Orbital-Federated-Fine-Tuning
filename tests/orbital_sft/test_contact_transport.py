import hashlib
from dataclasses import asdict

import pytest

from swarmshield.orbital.contact_transport import (
    allocate_slots, audit_slots, make_outages, predict_contacts, run_manifest,
)
from swarmshield.orbital.reliable_transfer import TransferSpec


def manifest_for(data):
    contacts = [{"client": i, "station": "shared", "start": 0.2, "end": 5.9, "delay_seconds": 0} for i in range(5)]
    slots = allocate_slots(contacts, 6, 0.15)
    return {"case": {"name": "synthetic-test"}, "base_identity": "synthetic@" + hashlib.sha256(data).hexdigest(),
            "transfers": [asdict(TransferSpec.from_bytes(f"client-{i}-v0", i, 0,
                "synthetic@" + hashlib.sha256(data).hexdigest(), data, 8192)) for i in range(5)],
            "contacts": contacts, "slots": slots, "outages": [], "deadline_seconds": 6,
            "socket_idle_timeout_seconds": 11, "condition": "contact-test", "queue_bytes": 1048576,
            "processing_guard_seconds": 0.03, "admission_lead_seconds": 0.003,
            "timing_tolerance_seconds": 0.10,
            "network": {"rate_bps": 1000000, "segment_bytes": 2048, "rtt_seconds": 0.001,
                        "retry_seconds": 0.02, "outage_seconds": 0}}


def test_shared_station_and_client_half_duplex_with_two_stations():
    contacts = [{"client": i, "station": station, "start": 0, "end": 2, "delay_seconds": 0}
                for i in range(5) for station in ("a", "b")]
    slots = allocate_slots(contacts, 2, 0.1)
    assert all(audit_slots({"slots": slots, "contacts": contacts, "deadline_seconds": 2}).values())
    assert any(len(s["competing_clients"]) > 1 for s in slots)
    for start in {s["start"] for s in slots}:
        simultaneous = [s for s in slots if s["start"] == start]
        assert len({s["client"] for s in simultaneous}) == len(simultaneous) == 2


def test_reservations_exclude_contact_edges_gaps_and_fault_oracle():
    contacts = [{"client": 0, "station": "a", "start": 0.15, "end": 0.95, "delay_seconds": 0},
                {"client": 0, "station": "a", "start": 1.25, "end": 1.95, "delay_seconds": 0}]
    slots = allocate_slots(contacts, 2, 0.2)
    frozen = [dict(s) for s in slots]
    first = make_outages(slots, 101, 0.2)
    second = make_outages(slots, 102, 0.2)
    assert first != second
    assert slots == frozen
    assert all(s["end"] <= 0.95 or s["start"] >= 1.25 for s in slots)
    assert make_outages(slots, None, 0.2) == []


def test_partial_payload_and_eligibility_before_confirmation_are_distinct():
    data = b"x" * 16384
    manifest = manifest_for(data)
    client_zero = [s for s in manifest["slots"] if s["client"] == 0]
    manifest["slots"] = [dict(s, id=i) for i, s in enumerate(client_zero[:4])]
    expected = predict_contacts(manifest)["clients"]["0"]
    assert expected["eligible"] and not expected["delivered"]
    manifest["slots"] = manifest["slots"][:2]
    expected = predict_contacts(manifest)["clients"]["0"]
    assert not expected["eligible"] and not expected["delivered"]


@pytest.mark.parametrize("value", [float("nan"), 0, -1])
def test_invalid_replay_clock_rejected(value):
    with pytest.raises(ValueError):
        allocate_slots([], 10, value)


@pytest.mark.parametrize("interrupted_index", [6, 21])
def test_actual_sockets_share_station_observe_outage_and_resume(tmp_path, interrupted_index):
    data = bytes(range(256)) * 64
    source = tmp_path / "synthetic-payload.bin"
    source.write_bytes(data)
    manifest = manifest_for(data)
    interrupted = manifest["slots"][interrupted_index]
    manifest["outages"] = [{"kind": "client", "target": interrupted["client"],
                             "start": interrupted["start"], "end": interrupted["end"]}]
    result = run_manifest(manifest, source, tmp_path / "contact-test")
    assert result["status"] == "passed", result["failed_checks"]
    assert result["contention_slots"] > 0
    assert sum(row["outage_slots"] for row in result["rows"]) == 1
    assert sum(row["measured_eligible"] for row in result["rows"]) == 5
