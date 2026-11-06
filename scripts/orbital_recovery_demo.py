import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from swarmshield.orbital.config import Network, digest
from swarmshield.orbital.network import LinkEngine
from swarmshield.orbital.reliable_transfer import DurableReceiver, TransferSpec


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/orbital/engineering-v1/recovery")
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    source = ROOT / "results/orbital/reference-22367142/round-0000/global_B.safetensors"
    data = source.read_bytes()
    base = "archived-B@" + digest(source)
    spec = TransferSpec.from_bytes("real-adapter-update", 0, 0, base, data)
    receiver = DurableReceiver(output / "receiver.sqlite", base)
    receiver.begin(spec)
    trace = []
    attempted = 0

    def record(event):
        trace.append({"event": event, "verified_bytes": spec.size - sum(min(spec.chunk_bytes, spec.size - i * spec.chunk_bytes) for i in receiver.missing(spec.transfer_id)),
                      "attempted_payload_bytes": attempted, "eligible_count": receiver.eligible_count()})

    block = data[:spec.chunk_bytes]
    attempted += len(block)
    receiver.put(spec.transfer_id, 0, block)
    record("first_verified_chunk")
    receiver.close()
    worker = """
import os, sys
from swarmshield.orbital.reliable_transfer import DurableReceiver
r = DurableReceiver(sys.argv[1], sys.argv[2])
data = open(sys.argv[3], 'rb').read()
r.put('real-adapter-update', 1, data[1048576:2097152])
os._exit(73)
"""
    crash = subprocess.run([sys.executable, "-c", worker, str(output / "receiver.sqlite"), base, str(source)], timeout=60)
    if crash.returncode != 73:
        raise AssertionError("Crash injection did not execute")
    attempted += spec.chunk_bytes
    receiver = DurableReceiver(output / "receiver.sqlite", base)
    if 1 in receiver.missing(spec.transfer_id):
        raise AssertionError("Acknowledgment-loss recovery lost committed chunk")
    record("process_exit_73_after_commit_before_ack")
    attempted += spec.chunk_bytes
    if receiver.put(spec.transfer_id, 1, data[spec.chunk_bytes:2 * spec.chunk_bytes]):
        raise AssertionError("Duplicate chunk was counted twice")
    record("lost_ack_retried_idempotently")
    corrupt = bytearray(data[2 * spec.chunk_bytes:3 * spec.chunk_bytes])
    corrupt[0] ^= 1
    attempted += len(corrupt)
    try:
        receiver.put(spec.transfer_id, 2, bytes(corrupt))
    except ValueError:
        record("corrupt_chunk_rejected")
    else:
        raise AssertionError("Corruption accepted")
    try:
        receiver.finalize(spec.transfer_id)
    except ValueError:
        record("incomplete_payload_not_eligible")
    else:
        raise AssertionError("Partial update became eligible")
    for i in reversed(receiver.missing(spec.transfer_id)):
        chunk = data[i * spec.chunk_bytes:(i + 1) * spec.chunk_bytes]
        attempted += len(chunk)
        receiver.put(spec.transfer_id, i, chunk)
        record(f"resumed_chunk_{i}")
    assert receiver.finalize(spec.transfer_id)["newly_eligible"]
    assert not receiver.finalize(spec.transfer_id)["newly_eligible"]
    record("full_hash_verified_single_eligibility_commit")
    receiver.advance(1)
    try:
        receiver.put(spec.transfer_id, 0, block)
    except ValueError:
        record("old_version_rejected_after_advance")
    else:
        raise AssertionError("Stale update accepted")
    receiver.close()
    contact_list = [{"client": 0, "station": "test-ground", "start": 0, "end": 18, "delay_seconds": 0},
                    {"client": 0, "station": "test-ground", "start": 100, "end": 200, "delay_seconds": 0}]
    network = Network(rate_bps=1000000, chunk_bytes=spec.chunk_bytes, framing_bytes=64, ack_bytes=32, acquisition_seconds=0)
    engine = LinkEngine(contact_list, network)
    receipt = engine.transfer(0, len(data), 0, 200, "upload", "interrupted-contact", source)
    assert receipt["complete"] and any(c["start"] >= 100 for c in receipt["chunks"])
    restored = LinkEngine(contact_list, network)
    restored.restore(json.loads(json.dumps(engine.snapshot())))
    uninterrupted_next = engine.transfer(0, 100, 0, 200, "download", "next")
    resumed_next = restored.transfer(0, 100, 0, 200, "download", "next")
    assert uninterrupted_next == resumed_next
    unavailable = LinkEngine([], network).transfer(0, len(data), 0, 200, "upload", "offline")
    assert not unavailable["complete"] and unavailable["verified_bytes"] == 0
    result = {"status": "passed", "input": str(source.relative_to(ROOT)), "input_sha256": digest(source),
              "payload_bytes": len(data), "attempted_payload_bytes": attempted, "trace": trace,
              "contact_interruption_receipt": receipt, "closed_boundary_reservation_restore_equal": True,
              "unavailable_client_not_delivered": True, "forced_process_exit_code": crash.returncode,
              "scope": "CPU durable-receiver extension and existing link-engine checks. Not integrated into archived GPU training or a socket/radio transport. Eligibility is transactional; model aggregation is a separate operation.",
              "limitations": ["SHA256 detects corruption against trusted manifests; it does not authenticate senders.",
                              "Process-death tests do not emulate filesystem/hardware power loss.",
                              "No optimizer or in-flight GPU-step recovery; mission-run automatic resume remains unsupported.",
                              "Attempted bytes include payload retries only, not RF framing or retransmission timing."]}
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in result.items() if k not in ("trace", "contact_interruption_receipt")}, indent=2))


if __name__ == "__main__":
    main()
