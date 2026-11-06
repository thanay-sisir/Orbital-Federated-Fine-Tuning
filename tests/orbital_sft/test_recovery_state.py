import hashlib
import sqlite3

import pytest

from swarmshield.orbital.recovery_state import FaultInjector, MissionLedger
from swarmshield.orbital.reliable_transfer import DurableReceiver, TransferSpec


def state(position, payload=b"weights"):
    return {"next_round": position, "global_sha256": hashlib.sha256(payload).hexdigest()}


def test_atomic_model_and_progress_commit_is_idempotent(tmp_path):
    path = tmp_path / "ledger.sqlite"
    ledger = MissionLedger(path, {"source": "one", "config": "same"})
    assert ledger.commit(-1, state(0), b"weights", {})
    assert ledger.commit(0, state(1), b"weights", {"participants": [0, 1]})
    assert not ledger.commit(0, state(1), b"weights", {"participants": [0, 1]})
    with pytest.raises(ValueError, match="Conflicting"):
        ledger.commit(0, state(1), b"weights", {"participants": [0]})
    ledger.close()
    ledger = MissionLedger(path, {"source": "one", "config": "same"})
    assert ledger.latest() == (state(1), b"weights")
    assert len(ledger.rounds()) == 1
    ledger.close()


def test_resume_rejects_changed_source_and_broken_model_blob(tmp_path):
    path = tmp_path / "ledger.sqlite"
    ledger = MissionLedger(path, {"source": "one"})
    ledger.commit(-1, state(0), b"weights", {})
    ledger.db.execute("UPDATE commits SET adapter=?", (b"corrupt",))
    ledger.db.commit()
    with pytest.raises(ValueError, match="blob checksum"):
        ledger.latest()
    ledger.close()
    with pytest.raises(ValueError, match="source"):
        MissionLedger(path, {"source": "different"})


def test_round_order_and_model_checksum_guards(tmp_path):
    ledger = MissionLedger(tmp_path / "ledger.sqlite", {})
    with pytest.raises(ValueError, match="sequential"):
        ledger.commit(0, state(1), b"weights", {})
    ledger.commit(-1, state(0), b"weights", {})
    with pytest.raises(ValueError, match="checksum"):
        ledger.commit(0, state(1), b"other", {})
    with pytest.raises(ValueError, match="sequential"):
        ledger.commit(1, state(2), b"weights", {})
    ledger.close()


def test_fault_plan_is_consumed_once_across_restarts(tmp_path):
    plan = [{"stage": "after_round_commit", "round": 1}]
    fault = FaultInjector(tmp_path / "faults.json", plan, "raise")
    fault.hit("after_round_commit", round=0)
    with pytest.raises(RuntimeError, match="Injected"):
        fault.hit("after_round_commit", round=1)
    FaultInjector(tmp_path / "faults.json", plan, "raise").hit("after_round_commit", round=1)


def test_receiver_export_requires_complete_verified_payload(tmp_path):
    receiver = DurableReceiver(tmp_path / "receiver.sqlite", "base")
    spec = TransferSpec.from_bytes("up", 0, 0, "base", b"abcdefgh", 4)
    receiver.begin(spec)
    receiver.put("up", 0, b"abcd")
    with pytest.raises(ValueError, match="finalized"):
        receiver.export("up", tmp_path / "received.bin")
    receiver.put("up", 1, b"efgh")
    receiver.finalize("up")
    receiver.export("up", tmp_path / "received.bin")
    assert (tmp_path / "received.bin").read_bytes() == b"abcdefgh"
    receiver.db.execute("UPDATE chunks SET data=? WHERE position=1", (b"xxxx",))
    receiver.db.commit()
    with pytest.raises(ValueError, match="corrupt"):
        receiver.export("up", tmp_path / "bad.bin")
    receiver.close()
