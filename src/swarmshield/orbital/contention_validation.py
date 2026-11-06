import hashlib
import json
import sqlite3
from pathlib import Path

from .socket_transport import specifications


def validate_payload_progress(execution, source, minimum_clients, minimum_bytes_per_client):
    execution = Path(execution)
    manifest = json.loads((execution / "manifest.json").read_text())
    prediction = json.loads((execution / "prediction.json").read_text())
    proxy = json.loads((execution / "proxy.json").read_text())
    specs = {s.client: s for s in specifications(manifest).values()}
    if not 2 <= minimum_clients <= len(specs) or minimum_bytes_per_client <= 0:
        raise ValueError("Require meaningful payload progress from multiple clients")
    source_data = Path(source).read_bytes()
    checksum = hashlib.sha256(source_data).hexdigest()
    checks = {"source_matches_trusted_manifests": all(s.payload_sha256 == checksum for s in specs.values()),
              "no_outages_in_control": not manifest["outages"],
              "one_shared_station": len({s["station"] for s in manifest["slots"]}) == 1,
              "actual_competing_nominal_clients": any(len(s["competing_clients"]) >= minimum_clients for s in manifest["slots"])}
    stored = {c: {} for c in specs}
    valid = True
    with sqlite3.connect((execution / "receiver.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as database:
        for identity, client, position, block in database.execute(
                "SELECT t.id,t.client,c.position,c.data FROM chunks c JOIN transfers t ON t.id=c.transfer_id ORDER BY t.client,c.position"):
            spec = specs.get(client)
            if spec is None or identity != spec.transfer_id or not 0 <= position < len(spec.chunk_sha256):
                valid = False
                continue
            expected = source_data[position * spec.chunk_bytes:(position + 1) * spec.chunk_bytes]
            valid = valid and block == expected and hashlib.sha256(block).hexdigest() == spec.chunk_sha256[position]
            stored[client][position] = len(block)
    checks["every_stored_chunk_matches_original_bytes_and_hash"] = valid
    rows = []
    payload_events = [e for e in proxy["events"] if e.get("op") == "put" and e.get("outcome") == "complete"]
    payload_clients = {e["client"] for e in payload_events}
    checks["multiple_clients_completed_payload_transactions"] = len(payload_clients) >= minimum_clients
    for client, spec in specs.items():
        missing = set(prediction["clients"][str(client)]["missing"])
        expected_positions = set(range(len(spec.chunk_sha256))) - missing
        expected_bytes = sum(min(spec.chunk_bytes, spec.size - i * spec.chunk_bytes) for i in expected_positions)
        measured_bytes = sum(stored[client].values())
        checks[f"client-{client}:predicted_verified_positions"] = set(stored[client]) == expected_positions
        checks[f"client-{client}:predicted_verified_payload_bytes"] = measured_bytes == expected_bytes
        checks[f"client-{client}:payload_transactions_match_storage"] = {e["position"] for e in payload_events if e["client"] == client} == set(stored[client])
        rows.append({"client": client, "predicted_verified_chunks": len(expected_positions),
                     "measured_verified_chunks": len(stored[client]), "predicted_verified_payload_bytes": expected_bytes,
                     "measured_verified_payload_bytes": measured_bytes,
                     "completed_payload_transactions": sum(e["client"] == client for e in payload_events)})
    progressing = sum(r["measured_verified_payload_bytes"] >= minimum_bytes_per_client for r in rows)
    checks["minimum_clients_with_verified_payload"] = progressing >= minimum_clients
    total = sum(r["measured_verified_payload_bytes"] for r in rows)
    squares = sum(r["measured_verified_payload_bytes"] ** 2 for r in rows)
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "clients": rows,
            "minimum_clients": minimum_clients, "minimum_bytes_per_client": minimum_bytes_per_client,
            "clients_meeting_payload_threshold": progressing, "total_verified_payload_bytes": total,
            "jain_verified_byte_share": total ** 2 / (len(rows) * squares) if squares else 0.0,
            "scope": "Read-only verification of actual stored adapter chunks and completed put transactions; status traffic cannot satisfy payload thresholds. Partial chunks are not accepted full updates."}
