import json
from dataclasses import asdict

import pytest

from swarmshield.orbital.contention_validation import validate_payload_progress
from swarmshield.orbital.reliable_transfer import DurableReceiver, TransferSpec


def fixture(tmp_path, with_chunks):
    source = tmp_path / "synthetic.bin"
    source.write_bytes(b"abcdefghijkl")
    specs = [TransferSpec.from_bytes(f"client-{i}", i, 0, "synthetic", source.read_bytes(), 4) for i in range(5)]
    manifest = {"transfers": [asdict(s) for s in specs], "outages": [],
                "slots": [{"station": "shared", "competing_clients": list(range(5))}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    missing = [1, 2] if with_chunks else [0, 1, 2]
    (tmp_path / "prediction.json").write_text(json.dumps({"clients": {str(i): {"missing": missing} for i in range(5)}}))
    events = [{"client": i, "op": "put", "outcome": "complete", "position": 0} for i in range(5)] if with_chunks else []
    (tmp_path / "proxy.json").write_text(json.dumps({"events": events}))
    receiver = DurableReceiver(tmp_path / "receiver.sqlite", "synthetic", 1024)
    for spec in specs:
        receiver.begin(spec)
        if with_chunks:
            receiver.put(spec.transfer_id, 0, b"abcd")
    receiver.close()
    return source


def test_status_only_run_cannot_pass_payload_control(tmp_path):
    source = fixture(tmp_path, False)
    result = validate_payload_progress(tmp_path, source, 5, 4)
    assert result["status"] == "failed"
    assert not result["checks"]["minimum_clients_with_verified_payload"]
    assert not result["checks"]["multiple_clients_completed_payload_transactions"]


def test_verified_partial_chunks_from_all_clients_pass(tmp_path):
    source = fixture(tmp_path, True)
    result = validate_payload_progress(tmp_path, source, 5, 4)
    assert result["status"] == "passed"
    assert result["total_verified_payload_bytes"] == 20
    assert result["jain_verified_byte_share"] == 1


def test_persisted_chunk_corruption_is_detected(tmp_path):
    source = fixture(tmp_path, True)
    receiver = DurableReceiver(tmp_path / "receiver.sqlite", "synthetic", 1024)
    with receiver.db:
        receiver.db.execute("UPDATE chunks SET data=? WHERE transfer_id=?", (b"xxxx", "client-2"))
    receiver.close()
    result = validate_payload_progress(tmp_path, source, 5, 4)
    assert result["status"] == "failed"
    assert not result["checks"]["every_stored_chunk_matches_original_bytes_and_hash"]


def test_prediction_disagreement_is_not_hidden_by_nonzero_progress(tmp_path):
    source = fixture(tmp_path, True)
    prediction = json.loads((tmp_path / "prediction.json").read_text())
    prediction["clients"]["3"]["missing"] = [2]
    (tmp_path / "prediction.json").write_text(json.dumps(prediction))
    result = validate_payload_progress(tmp_path, source, 5, 4)
    assert result["status"] == "failed"
    assert not result["checks"]["client-3:predicted_verified_positions"]
