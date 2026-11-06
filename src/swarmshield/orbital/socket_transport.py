import hashlib
import json
import multiprocessing as mp
import os
import socket
import socketserver
import struct
import threading
import time
import traceback
import zipfile
from collections import Counter
from pathlib import Path

from .reliable_transfer import DurableReceiver, TransferSpec


ROOT = Path(__file__).resolve().parents[3]
MAX_HEADER = 16384
MAX_BODY = 4 * 1024 * 1024


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def project_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("All paths must stay inside the project")
    return path


def header_bytes(message, body_size=0):
    header = json.dumps({**message, "body_bytes": body_size}, sort_keys=True, separators=(",", ":")).encode()
    if len(header) > MAX_HEADER or not 0 <= body_size <= MAX_BODY:
        raise ValueError("Frame exceeds protocol bounds")
    return struct.pack("!I", len(header)) + header


def receive_exact(connection, count):
    blocks = bytearray()
    while len(blocks) < count:
        block = connection.recv(count - len(blocks))
        if not block:
            raise EOFError("Connection ended inside frame")
        blocks.extend(block)
    return bytes(blocks)


def read_header(connection):
    prefix = receive_exact(connection, 4)
    size = struct.unpack("!I", prefix)[0]
    if not 1 <= size <= MAX_HEADER:
        raise ValueError("Invalid header length")
    raw = receive_exact(connection, size)
    message = json.loads(raw)
    size = message.get("body_bytes")
    if type(size) is not int or not 0 <= size <= MAX_BODY:
        raise ValueError("Invalid body length")
    return message, prefix + raw


def read_frame(connection):
    message, _ = read_header(connection)
    return message, receive_exact(connection, message["body_bytes"])


def send_frame(connection, message, body=b""):
    connection.sendall(header_bytes(message, len(body)) + body)


def connect(address, timeout):
    connection = socket.create_connection(address, timeout=timeout)
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return connection


def frame_timeout(manifest):
    size = max(s["chunk_bytes"] for s in manifest["transfers"])
    return max(manifest.get("socket_idle_timeout_seconds", 0),
               5 + (size + MAX_HEADER) * 8 / manifest["network"]["rate_bps"] + manifest["network"]["rtt_seconds"])


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = False
    daemon_threads = True

    def get_request(self):
        connection, address = super().get_request()
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        connection.settimeout(getattr(self, "frame_timeout", 5))
        return connection, address


def specifications(manifest):
    return {s["transfer_id"]: TransferSpec(**{**s, "chunk_sha256": tuple(s["chunk_sha256"])})
            for s in manifest["transfers"]}


def coordinator(output, manifest, epoch, go, stop, ready):
    output = Path(output)
    specs = specifications(manifest)
    database = output / "receiver.sqlite"
    initial = DurableReceiver(database, manifest["base_identity"], manifest["queue_bytes"])
    for spec in specs.values():
        initial.begin(spec)
    initial.close()
    lock = threading.Lock()
    events = []

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            receiver = None
            try:
                with lock:
                    receiver = DurableReceiver(database, manifest["base_identity"], manifest["queue_bytes"])
                while not stop.is_set():
                    message, body = read_frame(self.request)
                    identity = message.get("transfer_id")
                    if identity not in specs:
                        raise ValueError("Unknown trusted transfer identity")
                    spec = specs[identity]
                    operation = message.get("op")
                    with lock:
                        now = time.monotonic() - epoch.value
                        if now >= manifest["deadline_seconds"]:
                            send_frame(self.request, {"ok": False, "error": "deadline"})
                            return
                        if operation == "status" and not body:
                            missing = receiver.missing(identity)
                            response = {"ok": True, "missing": missing}
                        elif operation == "put":
                            position = message.get("position")
                            try:
                                stored = receiver.put(identity, position, body)
                                response = {"ok": True, "stored": stored}
                            except ValueError:
                                response = {"ok": False, "error": "checksum"}
                        elif operation == "finalize" and not body:
                            finalized = receiver.finalize(identity)
                            response = {"ok": True, "new": finalized["newly_eligible"], "sha256": finalized["payload_sha256"]}
                            receiver.export(identity, output / "received" / f"client-{spec.client}.safetensors")
                        else:
                            raise ValueError("Invalid operation")
                        events.append({"at": time.monotonic() - epoch.value, "client": spec.client,
                                       "op": operation, "position": message.get("position"), "response": response})
                    send_frame(self.request, response)
            except (EOFError, OSError):
                pass
            except Exception as exc:
                with lock:
                    events.append({"error": type(exc).__name__, "detail": str(exc)})
            finally:
                if receiver is not None:
                    receiver.close()

    with Server(("127.0.0.1", 0), Handler) as server:
        server.frame_timeout = frame_timeout(manifest)
        server.timeout = 0.1
        ready.put({"role": "coordinator", "address": server.server_address, "pid": os.getpid()})
        go.wait()
        while not stop.is_set():
            server.handle_request()
    receiver = DurableReceiver(database, manifest["base_identity"], manifest["queue_bytes"])
    save(output / "coordinator.json", {"pid": os.getpid(), "events": events,
         "eligible_count": receiver.eligible_count(),
         "persisted_payload_bytes": receiver.db.execute("SELECT COALESCE(SUM(LENGTH(data)),0) FROM chunks").fetchone()[0]})
    receiver.close()


def proxy(output, manifest, backend, epoch, go, stop, ready):
    output = Path(output)
    specs = specifications(manifest)
    network = manifest["network"]
    locks = {s.client: threading.Lock() for s in specs.values()}
    counters = {s.client: Counter() for s in specs.values()}
    blocked = {s.client: 0.0 for s in specs.values()}
    applied = set()
    events = []

    def forward(connection, data, client, direction):
        for offset in range(0, len(data), network["segment_bytes"]):
            block = data[offset:offset + network["segment_bytes"]]
            time.sleep(len(block) * 8 / network["rate_bps"])
            connection.sendall(block)
            counters[client][direction + "_bytes"] += len(block)

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                message, raw = read_header(self.request)
                spec = specs[message["transfer_id"]]
                client = spec.client
                with locks[client]:
                    now = time.monotonic() - epoch.value
                    counters[client]["connections"] += 1
                    if (manifest["condition"] == "deadline" and client == 4) or now < blocked[client]:
                        counters[client]["unavailable_connections"] += 1
                        return
                    with connect(backend, frame_timeout(manifest)) as upstream:
                        while not stop.is_set():
                            if message["transfer_id"] != spec.transfer_id:
                                raise ValueError("Connection changed client identity")
                            operation = message.get("op")
                            position = message.get("position")
                            mode = None
                            if manifest["condition"] == "recovery" and operation == "put":
                                if client == 0 and position == 1 and "cut" not in applied:
                                    mode = "cut"
                                elif client == 1 and position == 0 and "lost_ack" not in applied:
                                    mode = "lost_ack"
                                elif client == 2 and position == 0 and "corruption" not in applied:
                                    mode = "corruption"
                            body_size = message["body_bytes"]
                            body = receive_exact(self.request, body_size)
                            if mode:
                                applied.add(mode)
                                events.append({"at": time.monotonic() - epoch.value, "client": client,
                                               "event": mode, "position": position})
                            time.sleep(network["rtt_seconds"] / 2)
                            forward(upstream, raw, client, "request")
                            if mode == "cut":
                                forward(upstream, body[:body_size // 2], client, "request")
                                blocked[client] = time.monotonic() - epoch.value + network["outage_seconds"]
                                counters[client]["interrupted_prefix_bytes"] += body_size // 2
                                return
                            if mode == "corruption":
                                body = bytes([body[0] ^ 1]) + body[1:]
                            forward(upstream, body, client, "request")
                            response, response_raw = read_header(upstream)
                            response_body = receive_exact(upstream, response["body_bytes"])
                            if mode == "lost_ack":
                                blocked[client] = time.monotonic() - epoch.value + network["outage_seconds"]
                                counters[client]["dropped_response_bytes"] += len(response_raw) + len(response_body)
                                return
                            time.sleep(network["rtt_seconds"] / 2)
                            forward(self.request, response_raw + response_body, client, "response")
                            message, raw = read_header(self.request)
            except (EOFError, OSError):
                pass
            except Exception as exc:
                events.append({"error": type(exc).__name__, "detail": str(exc)})

    with Server(("127.0.0.1", 0), Handler) as server:
        server.frame_timeout = frame_timeout(manifest)
        server.timeout = 0.1
        ready.put({"role": "proxy", "address": server.server_address, "pid": os.getpid()})
        go.wait()
        while not stop.is_set():
            server.handle_request()
    save(output / "proxy.json", {"pid": os.getpid(), "events": events,
                                 "clients": {str(k): dict(v) for k, v in counters.items()}})


def client_worker(output, manifest, spec_dict, source, address, epoch, go, ready):
    spec = TransferSpec(**{**spec_dict, "chunk_sha256": tuple(spec_dict["chunk_sha256"])})
    ready.put({"role": f"client-{spec.client}", "pid": os.getpid()})
    go.wait()
    deadline = epoch.value + manifest["deadline_seconds"]
    events = []
    result = {"client": spec.client, "pid": os.getpid(), "delivered": False}
    while time.monotonic() < deadline:
        try:
            with connect(address, min(3, max(0.001, deadline - time.monotonic()))) as connection:
                def request(operation, body=b"", **extra):
                    connection.settimeout(min(frame_timeout(manifest), max(0.001, deadline - time.monotonic())))
                    send_frame(connection, {"op": operation, "transfer_id": spec.transfer_id, **extra}, body)
                    response, _ = read_frame(connection)
                    return response

                status = request("status")
                if not status.get("ok"):
                    raise ConnectionError(str(status))
                events.append({"at": time.monotonic() - epoch.value, "missing": status["missing"]})
                with open(source, "rb") as stream:
                    for position in status["missing"]:
                        stream.seek(position * spec.chunk_bytes)
                        body = stream.read(min(spec.chunk_bytes, spec.size - position * spec.chunk_bytes))
                        if not request("put", body, position=position).get("ok"):
                            raise ConnectionError("Chunk rejected")
                first = request("finalize")
                second = request("finalize")
                if not first.get("ok") or not second.get("ok") or second.get("new") is not False:
                    raise ValueError("Finalization was not idempotent")
                if first["sha256"] != spec.payload_sha256 or second["sha256"] != spec.payload_sha256:
                    raise ValueError("Final checksum differs")
                result.update(delivered=True, completed_seconds=time.monotonic() - epoch.value)
                break
        except (OSError, EOFError) as exc:
            events.append({"at": time.monotonic() - epoch.value, "retry": type(exc).__name__})
            time.sleep(min(manifest["network"]["retry_seconds"], max(0, deadline - time.monotonic())))
    result["events"] = events
    result["elapsed_seconds"] = time.monotonic() - epoch.value
    save(Path(output) / f"client-{spec.client}.json", result)


def predict(spec, network, condition):
    if condition == "deadline" and spec.client == 4:
        return {"delivered": False, "request_bytes": 0, "response_bytes": 0, "predicted_seconds": None}
    request_bytes = response_bytes = 0
    propagation = hold = 0.0
    missing = list(range(len(spec.chunk_sha256)))

    def exchange(operation, response, size=0, position=None, prefix=None, drop_response=False):
        nonlocal request_bytes, response_bytes, propagation
        request = {"op": operation, "transfer_id": spec.transfer_id}
        if position is not None:
            request["position"] = position
        request_bytes += len(header_bytes(request, size)) + (size if prefix is None else prefix)
        propagation += network["rtt_seconds"] / 2
        if prefix is None and not drop_response:
            response_bytes += len(header_bytes(response))
            propagation += network["rtt_seconds"] / 2

    exchange("status", {"ok": True, "missing": missing.copy()})
    for position in range(len(spec.chunk_sha256)):
        size = min(spec.chunk_bytes, spec.size - position * spec.chunk_bytes)
        if condition == "recovery" and spec.client == 0 and position == 1:
            exchange("put", None, size, position, prefix=size // 2)
            hold += network["outage_seconds"] + network["retry_seconds"] / 2
            exchange("status", {"ok": True, "missing": missing.copy()})
        if condition == "recovery" and spec.client == 2 and position == 0:
            exchange("put", {"ok": False, "error": "checksum"}, size, position)
            hold += network["retry_seconds"]
            exchange("status", {"ok": True, "missing": missing.copy()})
        if condition == "recovery" and spec.client == 1 and position == 0:
            exchange("put", None, size, position, drop_response=True)
            missing.remove(position)
            hold += network["outage_seconds"] + network["retry_seconds"] / 2
            exchange("status", {"ok": True, "missing": missing.copy()})
            continue
        exchange("put", {"ok": True, "stored": True}, size, position)
        missing.remove(position)
    for new in (True, False):
        exchange("finalize", {"ok": True, "new": new, "sha256": spec.payload_sha256})
    serialization = (request_bytes + response_bytes) * 8 / network["rate_bps"]
    return {"delivered": True, "request_bytes": request_bytes, "response_bytes": response_bytes,
            "serialization_seconds": serialization, "propagation_seconds": propagation,
            "hold_seconds": hold, "predicted_seconds": serialization + propagation + hold}


def entry(role, output, target, args):
    try:
        target(*args)
    except BaseException:
        save(Path(output) / f"{role}.error.json", {"traceback": traceback.format_exc()})
        raise


def execute_case(output, manifest, source, proxy_target=proxy):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "received").mkdir()
    save(output / "manifest.json", manifest)
    context = mp.get_context("spawn")
    epoch = context.Value("d", 0)
    go, stop, ready = context.Event(), context.Event(), context.Queue()
    processes = []
    roles = []

    def start(role, target, args):
        process = context.Process(target=entry, args=(role, str(output), target, args), name=role)
        process.start()
        processes.append(process)
        record = ready.get(timeout=30)
        if record["role"] != role:
            raise RuntimeError("Unexpected process readiness")
        roles.append(record)
        return record

    try:
        backend = start("coordinator", coordinator, (str(output), manifest, epoch, go, stop, ready))
        gateway = start("proxy", proxy_target, (str(output), manifest, backend["address"], epoch, go, stop, ready))
        for spec in manifest["transfers"]:
            start(f"client-{spec['client']}", client_worker,
                  (str(output), manifest, spec, str(source), gateway["address"], epoch, go, ready))
        epoch.value = time.monotonic()
        go.set()
        for process in processes[2:]:
            process.join(max(0, epoch.value + manifest["deadline_seconds"] + 8 - time.monotonic()))
            if process.is_alive() or process.exitcode != 0:
                raise RuntimeError(f"Client {process.name} failed or timed out")
        stop.set()
        for process in processes[:2]:
            process.join(8)
            if process.is_alive() or process.exitcode != 0:
                raise RuntimeError(f"Service {process.name} failed or timed out")
        save(output / "processes.json", roles)
    finally:
        stop.set()
        go.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(5)
        ready.close()
        ready.join_thread()


def validate_config(config):
    if config["clients"] != 5:
        raise ValueError("This experiment requires five client processes")
    for name in ("rate_bps", "segment_bytes", "chunk_bytes", "queue_bytes"):
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f"Invalid {name}")
    if not 1024 <= config["chunk_bytes"] <= MAX_BODY or config["segment_bytes"] > config["chunk_bytes"]:
        raise ValueError("Invalid chunk or forwarding segment size")
    for name in ("rtt_seconds", "outage_seconds", "retry_seconds", "deadline_seconds",
                 "timing_relative_tolerance", "timing_absolute_tolerance_seconds"):
        if type(config[name]) not in (int, float) or not 0 <= config[name] < 3600:
            raise ValueError(f"Invalid {name}")
    if config["deadline_seconds"] <= 0 or config["retry_seconds"] <= 0:
        raise ValueError("Positive deadline and retry interval required")


def run_experiment(config, output):
    validate_config(config)
    output = project_path(output)
    source = project_path(config["payload"])
    data = source.read_bytes()
    if len(data) <= config["chunk_bytes"]:
        raise ValueError("Payload must contain at least two chunks")
    if len(data) * 5 > config["queue_bytes"]:
        raise ValueError("Five payloads exceed the declared storage bound")
    base = "opaque-archived-adapter@" + hashlib.sha256(data).hexdigest()
    specs = [TransferSpec.from_bytes(f"client-{i}-v0", i, 0, base, data, config["chunk_bytes"]) for i in range(5)]
    network = {k: config[k] for k in ("rate_bps", "segment_bytes", "rtt_seconds", "outage_seconds", "retry_seconds")}
    output.mkdir(parents=True, exist_ok=False)
    save(output / "config.json", config)
    source_files = [Path(__file__), Path(__file__).with_name("reliable_transfer.py"),
                    ROOT / "scripts/orbital_socket_demo.py", ROOT / "scripts/orbital_socket_plots.py",
                    ROOT / "tests/orbital_sft/test_socket_transport.py", ROOT / "tests/orbital_sft/test_reliable_transfer.py",
                    ROOT / "infra/rivanna/socket_transport.sbatch", ROOT / "infra/rivanna/common.sh",
                    ROOT / "infra/rivanna/environment/requirements-analysis.lock.txt"]
    source_files = [project_path(path) for path in source_files]
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in source_files:
            archive.write(path, str(path.relative_to(ROOT)))
        archive.writestr("resolved_config.json", json.dumps(config, indent=2))
    save(output / "provenance.json", {"input": str(source.relative_to(ROOT)), "payload_bytes": len(data),
         "input_sha256": hashlib.sha256(data).hexdigest(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
         "snapshot_sha256": hashlib.sha256((output / "source_snapshot.zip").read_bytes()).hexdigest(),
         "python": os.sys.version, "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}})
    from dataclasses import asdict
    for condition in ("control", "recovery", "deadline"):
        manifest = {"condition": condition, "base_identity": base, "queue_bytes": config["queue_bytes"],
                    "deadline_seconds": config["deadline_seconds"], "network": network,
                    "transfers": [asdict(spec) for spec in specs]}
        predictions = [predict(spec, network, condition) for spec in specs]
        if max(p["predicted_seconds"] or 0 for p in predictions) + 2 >= config["deadline_seconds"]:
            raise ValueError("Deadline must leave at least two seconds beyond predicted recoverable delivery")
        save(output / f"predictions-{condition}.json", predictions)
        execute_case(output / condition, manifest, source)
    return summarize(output)


def summarize(output):
    output = Path(output)
    config = json.loads((output / "config.json").read_text())
    rows, checks = [], {}
    for condition in ("control", "recovery", "deadline"):
        case = output / condition
        coordinator_result = json.loads((case / "coordinator.json").read_text())
        proxy_result = json.loads((case / "proxy.json").read_text())
        manifest = json.loads((case / "manifest.json").read_text())
        specs = specifications(manifest)
        predictions = json.loads((output / f"predictions-{condition}.json").read_text())
        checks[condition + ":services_no_errors"] = not any("error" in e for e in coordinator_result["events"] + proxy_result["events"])
        checks[condition + ":five_distinct_clients_and_two_services"] = len({p["pid"] for p in json.loads((case / "processes.json").read_text())}) == 7
        completed = 0
        for client in range(5):
            measured = json.loads((case / f"client-{client}.json").read_text())
            predicted = predictions[client]
            counts = proxy_result["clients"][str(client)]
            spec = specs[f"client-{client}-v0"]
            key = f"{condition}:client-{client}:"
            checks[key + "delivery"] = measured["delivered"] == predicted["delivered"]
            checks[key + "request_bytes"] = counts.get("request_bytes", 0) == predicted["request_bytes"]
            checks[key + "response_bytes"] = counts.get("response_bytes", 0) == predicted["response_bytes"]
            events = [e for e in coordinator_result["events"] if e.get("client") == client]
            newly_eligible = sum(e.get("response", {}).get("new") is True for e in events)
            checks[key + "unique_eligibility"] = newly_eligible == int(predicted["delivered"])
            if measured["delivered"]:
                completed += 1
                exported = (case / "received" / f"client-{client}.safetensors").read_bytes()
                checks[key + "export_hash"] = hashlib.sha256(exported).hexdigest() == spec.payload_sha256
                checks[key + "before_deadline"] = measured["completed_seconds"] < config["deadline_seconds"]
                error = measured["completed_seconds"] - predicted["predicted_seconds"]
                tolerance = config["timing_absolute_tolerance_seconds"] + predicted["predicted_seconds"] * config["timing_relative_tolerance"]
                checks[key + "timing_envelope"] = abs(error) <= tolerance
            else:
                error = None
                checks[key + "no_export"] = not (case / "received" / f"client-{client}.safetensors").exists()
            rows.append({"condition": condition, "client": client, "delivered": measured["delivered"],
                         "measured_seconds": measured.get("completed_seconds"), "predicted_seconds": predicted["predicted_seconds"],
                         "timing_error_seconds": error, "measured_request_bytes": counts.get("request_bytes", 0),
                         "predicted_request_bytes": predicted["request_bytes"], "measured_response_bytes": counts.get("response_bytes", 0),
                         "predicted_response_bytes": predicted["response_bytes"], "connections": counts.get("connections", 0)})
        checks[condition + ":eligible_count"] = coordinator_result["eligible_count"] == completed
        checks[condition + ":storage_bound"] = coordinator_result["persisted_payload_bytes"] <= config["queue_bytes"]
        if condition == "recovery":
            checks["recovery:all_three_faults_injected"] = {e.get("event") for e in proxy_result["events"]} == {"cut", "lost_ack", "corruption"}
            events = coordinator_result["events"]
            for client in (0, 1):
                puts = [e for e in events if e.get("client") == client and e.get("op") == "put" and e.get("position") == 0]
                checks[f"recovery:client-{client}:verified_chunk_not_retransmitted"] = len(puts) == 1
                statuses = [e for e in events if e.get("client") == client and e.get("op") == "status"]
                checks[f"recovery:client-{client}:resume_observed"] = len(statuses) == 2 and 0 not in statuses[1]["response"]["missing"]
            checks["recovery:corruption_rejected"] = sum(e.get("response", {}).get("error") == "checksum" for e in events) == 1
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks, "rows": rows,
              "scope": "Measured loopback TCP application-frame transport, five client processes, one coordinator and one protocol-aware proxy. Saved adapter replay; no model loading, training or aggregation.",
              "prediction_scope": "Pre-execution protocol serialization/latency/outage model, not a replay of the orbital LinkEngine. Poll phase approximated by half a retry interval. CPU, filesystem and kernel overhead are measured residuals.",
              "limits": ["One host, five independent per-client rate limits; no shared station contention, orbital geometry or flight hardware.",
                         "Byte counts cover application frames forwarded by the proxy, excluding TCP/IP overhead, dropped ACK bytes on the backend leg and bytes buffered on the ingress leg.",
                         "Proxy buffers at most one bounded chunk per handler; this is a protocol-aware emulator, not packet-level RF emulation.",
                         "Faults trigger at declared chunk boundaries; client 4 is unavailable for the entire deadline case.",
                         "Same archived adapter is replayed by all five clients; these are not five new learned updates.",
                         "Trusted manifest checksums detect corruption; no sender authentication or cryptographic transport security."]}
    save(output / "results.json", result)
    lines = ["# Measured socket-transfer validation", "", f"Status: **{result['status']}**; {sum(checks.values())}/{len(checks)} checks passed.", "", result["scope"], "", result["prediction_scope"], "",
             "| Condition | Client | Delivered | Predicted s | Measured s | Timing error s | Forwarded request bytes |", "|---|---:|---|---:|---:|---:|---:|"]
    for row in rows:
        def number(value):
            return "—" if value is None else f"{value:.3f}"
        lines.append(f"| {row['condition']} | {row['client']} | {row['delivered']} | {number(row['predicted_seconds'])} | {number(row['measured_seconds'])} | {number(row['timing_error_seconds'])} | {row['measured_request_bytes']} |")
    lines.extend(["", "## Failed checks", "", *([f"- {key}" for key, value in checks.items() if not value] or ["None."]), "", "## Limits", "", *[f"- {limit}" for limit in result["limits"]]])
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return result
