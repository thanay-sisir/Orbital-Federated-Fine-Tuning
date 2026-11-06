import json
import os
import socket
import struct
import subprocess
import sys
import zipfile

import pytest

from swarmshield.orbital.socket_transport import (
    MAX_BODY, MAX_HEADER, ROOT, header_bytes, read_frame,
    run_experiment, validate_config,
)


def configuration(payload):
    return {"payload": str(payload), "clients": 5, "rate_bps": 1000000,
            "chunk_bytes": 8192, "segment_bytes": 2048, "rtt_seconds": 0.001,
            "outage_seconds": 0.15, "retry_seconds": 0.02, "queue_bytes": 1048576,
            "deadline_seconds": 4, "timing_relative_tolerance": 0.5,
            "timing_absolute_tolerance_seconds": 1}


@pytest.mark.parametrize("frame", [struct.pack("!I", MAX_HEADER + 1),
                                   struct.pack("!I", 0), header_bytes({"op": "put"})[:-1]])
def test_malformed_or_truncated_frame_is_rejected(frame):
    left, right = socket.socketpair()
    try:
        left.sendall(frame)
        left.shutdown(socket.SHUT_WR)
        with pytest.raises((ValueError, EOFError)):
            read_frame(right)
    finally:
        left.close()
        right.close()


def test_oversized_body_rejected_before_payload_read():
    left, right = socket.socketpair()
    raw = json.dumps({"body_bytes": MAX_BODY + 1}).encode()
    try:
        left.sendall(struct.pack("!I", len(raw)) + raw)
        with pytest.raises(ValueError, match="body"):
            read_frame(right)
    finally:
        left.close()
        right.close()


@pytest.mark.parametrize("field,value", [("clients", 4), ("rate_bps", 0),
                                        ("chunk_bytes", MAX_BODY + 1), ("retry_seconds", 0),
                                        ("rtt_seconds", float("nan"))])
def test_invalid_experiment_budget_is_rejected(field, value):
    config = configuration("unused")
    config[field] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_five_process_transport_faults_predictions_and_deadline(tmp_path):
    payload = tmp_path / "synthetic-test-payload.bin"
    payload.write_bytes(bytes(range(256)) * 96)
    output = tmp_path / "run"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(configuration(payload)))
    alias = tmp_path / "source-alias"
    alias.symlink_to(ROOT / "src", target_is_directory=True)
    environment = {**os.environ, "PYTHONPATH": str(alias)}
    invocation = subprocess.run([sys.executable, str(ROOT / "scripts/orbital_socket_demo.py"),
                                 "--config", str(config_path), "--output", str(output), "--allow-local-cpu"],
                                env=environment, capture_output=True, text=True, timeout=40)
    assert invocation.returncode == 0, invocation.stdout + invocation.stderr
    result = json.loads((output / "results.json").read_text())
    with zipfile.ZipFile(output / "source_snapshot.zip") as archive:
        assert "src/swarmshield/orbital/socket_transport.py" in archive.namelist()
        assert all(not name.startswith("/") and "source-alias" not in name for name in archive.namelist())
    assert result["status"] == "passed", {k: v for k, v in result["checks"].items() if not v}
    assert len(result["rows"]) == 15
    assert sum(row["delivered"] for row in result["rows"]) == 14
    assert (output / "deadline/received/client-4.safetensors").exists() is False
    for condition in ("control", "recovery"):
        for client in range(5):
            assert (output / condition / "received" / f"client-{client}.safetensors").read_bytes() == payload.read_bytes()
    with pytest.raises(FileExistsError):
        run_experiment(configuration(payload), output)


def test_storage_bound_rejected_before_launch(tmp_path):
    payload = tmp_path / "synthetic-test-payload.bin"
    payload.write_bytes(b"x" * 24576)
    config = configuration(payload)
    config["queue_bytes"] = 24576
    with pytest.raises(ValueError, match="storage bound"):
        run_experiment(config, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()
