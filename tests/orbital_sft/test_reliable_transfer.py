import subprocess
import sys
from dataclasses import replace

import pytest

from swarmshield.orbital.reliable_transfer import DurableReceiver, TransferSpec


def setup(tmp_path, data=b"abcdefghij"):
    spec = TransferSpec.from_bytes("update", 0, 0, "base@revision", data, 4)
    receiver = DurableReceiver(tmp_path / "receiver.sqlite", "base@revision", 100)
    receiver.begin(spec)
    return receiver, spec, data


def test_corrupt_missing_duplicate_out_of_order_and_reopen(tmp_path):
    receiver, spec, data = setup(tmp_path)
    with pytest.raises(ValueError, match="checksum"):
        receiver.put(spec.transfer_id, 0, b"XXXX")
    assert receiver.put(spec.transfer_id, 2, data[8:])
    assert not receiver.put(spec.transfer_id, 2, data[8:])
    with pytest.raises(ValueError, match="Incomplete"):
        receiver.finalize(spec.transfer_id)
    receiver.close()
    receiver = DurableReceiver(tmp_path / "receiver.sqlite", "base@revision", 100)
    assert receiver.missing(spec.transfer_id) == [0, 1]
    receiver.put(spec.transfer_id, 0, data[:4])
    receiver.put(spec.transfer_id, 1, data[4:8])
    assert receiver.finalize(spec.transfer_id)["newly_eligible"]
    assert not receiver.finalize(spec.transfer_id)["newly_eligible"]
    assert receiver.eligible_count() == 1
    receiver.close()


def test_stale_conflicting_identity_wrong_base_and_second_update_rejected(tmp_path):
    receiver, spec, data = setup(tmp_path)
    with pytest.raises(ValueError, match="different metadata"):
        receiver.begin(replace(spec, client=1))
    with pytest.raises(ValueError, match="incompatible"):
        receiver.begin(replace(spec, transfer_id="wrong-base", base_identity="different"))
    for i in receiver.missing(spec.transfer_id):
        receiver.put(spec.transfer_id, i, data[i * 4:(i + 1) * 4])
    receiver.finalize(spec.transfer_id)
    second = replace(spec, transfer_id="second")
    receiver.begin(second)
    for i in receiver.missing(second.transfer_id):
        receiver.put(second.transfer_id, i, data[i * 4:(i + 1) * 4])
    with pytest.raises(ValueError, match="already has"):
        receiver.finalize(second.transfer_id)
    receiver.advance(1)
    with pytest.raises(ValueError, match="Stale"):
        receiver.put(second.transfer_id, 0, data[:4])
    with pytest.raises(ValueError, match="Stale"):
        receiver.finalize(second.transfer_id)
    with pytest.raises(ValueError, match="rollback"):
        receiver.advance(0)
    receiver.close()


def test_queue_reservation_and_full_payload_checksum(tmp_path):
    receiver, spec, data = setup(tmp_path)
    with pytest.raises(ValueError, match="capacity"):
        receiver.begin(TransferSpec.from_bytes("oversized", 1, 0, "base@revision", b"a" * 91, 4))
    wrong = replace(spec, transfer_id="full-hash", client=1, payload_sha256="0" * 64)
    receiver.begin(wrong)
    for i in receiver.missing(wrong.transfer_id):
        receiver.put(wrong.transfer_id, i, data[i * 4:(i + 1) * 4])
    with pytest.raises(ValueError, match="checksum"):
        receiver.finalize(wrong.transfer_id)
    receiver.close()


@pytest.mark.parametrize("after_finalize", [False, True])
def test_process_death_after_commit_before_ack(tmp_path, after_finalize):
    receiver, spec, data = setup(tmp_path)
    receiver.close()
    worker = """
import os, sys
from swarmshield.orbital.reliable_transfer import DurableReceiver
r = DurableReceiver(sys.argv[1], 'base@revision', 100)
r.put('update', 0, b'abcd')
if sys.argv[2] == 'True':
    r.put('update', 1, b'efgh')
    r.put('update', 2, b'ij')
    r.finalize('update')
os._exit(73)
"""
    result = subprocess.run([sys.executable, "-c", worker, str(tmp_path / "receiver.sqlite"), str(after_finalize)], timeout=30)
    assert result.returncode == 73
    receiver = DurableReceiver(tmp_path / "receiver.sqlite", "base@revision", 100)
    assert not receiver.put("update", 0, b"abcd")
    if after_finalize:
        assert receiver.missing("update") == []
        assert not receiver.finalize("update")["newly_eligible"]
        assert receiver.eligible_count() == 1
    else:
        assert receiver.missing("update") == [1, 2]
        assert receiver.eligible_count() == 0
    receiver.close()
