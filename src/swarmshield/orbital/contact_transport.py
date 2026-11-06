import hashlib
import json
import math
import os
import random
import socketserver
import threading
import time
import zipfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .reliable_transfer import TransferSpec
from .socket_transport import (
    ROOT, Server, connect, execute_case, frame_timeout, header_bytes,
    project_path, read_header, receive_exact, save, specifications,
)


def allocate_slots(contacts, deadline, slot_seconds):
    if not math.isfinite(deadline) or not math.isfinite(slot_seconds) or not 0 < slot_seconds <= deadline:
        raise ValueError("Invalid contact replay clock")
    stations = sorted({c["station"] for c in contacts})
    for contact in contacts:
        if not all(math.isfinite(contact[k]) for k in ("start", "end", "delay_seconds")):
            raise ValueError("Non-finite contact")
        if contact["start"] < 0 or contact["end"] <= contact["start"] or contact["delay_seconds"] < 0:
            raise ValueError("Invalid contact interval")
    slots, last = [], {s: -1 for s in stations}
    for tick in range(int(deadline / slot_seconds)):
        start, end = tick * slot_seconds, (tick + 1) * slot_seconds
        clients = set()
        for station in stations:
            available = [(i, c) for i, c in enumerate(contacts) if c["station"] == station
                         and c["start"] <= start and end <= c["end"] and c["client"] not in clients]
            if not available:
                continue
            available.sort(key=lambda pair: (pair[1]["client"] <= last[station], pair[1]["client"], pair[0]))
            index, chosen = available[0]
            last[station] = chosen["client"]
            clients.add(chosen["client"])
            slots.append({"id": len(slots), "client": chosen["client"], "station": station,
                          "start": start, "end": end, "contact": index,
                          "delay_seconds": chosen["delay_seconds"],
                          "competing_clients": sorted({c["client"] for _, c in available})})
    return slots


def make_outages(slots, seed, slot_seconds):
    if seed is None:
        return []
    rng = random.Random(seed)
    outages = []
    groups = [("station", s) for s in sorted({x["station"] for x in slots})]
    groups += [("client", c) for c in sorted({x["client"] for x in slots})]
    for kind, identity in groups:
        candidates = [x for x in slots if x[kind] == identity]
        if not candidates:
            continue
        for chosen in rng.sample(candidates, min(2, len(candidates))):
            outages.append({"kind": kind, "target": identity, "start": chosen["start"],
                            "end": chosen["start"] + rng.randint(1, 3) * slot_seconds})
    return sorted(outages, key=lambda o: (o["start"], o["kind"], str(o["target"])))


def outage_at(slot, outages):
    return any(o["start"] <= slot["start"] < o["end"] - 1e-9
               and ((o["kind"] == "client" and o["target"] == slot["client"])
                    or (o["kind"] == "station" and o["target"] == slot["station"])) for o in outages)


def predict_contacts(manifest):
    network = manifest["network"]
    states = {s.client: {"phase": "status", "missing": list(range(len(s.chunk_sha256))),
                        "request_bytes": 0, "response_bytes": 0, "failed_slots": 0,
                        "completed_seconds": None, "eligible": False, "transactions": 0} for s in specifications(manifest).values()}
    specs = {s.client: s for s in specifications(manifest).values()}
    trace = []
    for slot in manifest["slots"]:
        client = slot["client"]
        state, spec = states[client], specs[client]
        if state["phase"] == "done":
            continue
        if outage_at(slot, manifest["outages"]):
            state["phase"] = "status"
            state["failed_slots"] += 1
            trace.append({"slot": slot["id"], "client": client, "outcome": "outage"})
            continue
        phase = state["phase"]
        request = {"op": "finalize" if phase.startswith("finalize") else phase,
                   "transfer_id": spec.transfer_id}
        size = 0
        if phase == "status":
            response = {"ok": True, "missing": state["missing"].copy()}
            state["phase"] = "put" if state["missing"] else "finalize-first"
        elif phase == "put":
            position = state["missing"].pop(0)
            request["position"] = position
            size = min(spec.chunk_bytes, spec.size - position * spec.chunk_bytes)
            response = {"ok": True, "stored": True}
            state["phase"] = "put" if state["missing"] else "finalize-first"
        else:
            response = {"ok": True, "new": phase == "finalize-first" and not state["eligible"], "sha256": spec.payload_sha256}
            state["eligible"] = True
            state["phase"] = "finalize-second" if phase == "finalize-first" else "done"
        upstream = len(header_bytes(request, size)) + size
        downstream = len(header_bytes(response))
        duration = (upstream + downstream) * 8 / network["rate_bps"] + network["rtt_seconds"] + 2 * slot["delay_seconds"]
        if duration + manifest["processing_guard_seconds"] > slot["end"] - slot["start"] + 1e-9:
            raise ValueError("Nominal slot cannot fit its frame and declared processing guard")
        state["request_bytes"] += upstream
        state["response_bytes"] += downstream
        state["transactions"] += 1
        trace.append({"slot": slot["id"], "client": client, "op": request["op"],
                      "position": request.get("position"), "outcome": "complete"})
        if state["phase"] == "done":
            state["completed_seconds"] = slot["start"] + duration
    return {"clients": {str(k): {**v, "delivered": v["phase"] == "done"} for k, v in states.items()}, "trace": trace}


class RadioClosed(ConnectionError):
    pass


def contact_proxy(output, manifest, backend, epoch, go, stop, ready):
    output = Path(output)
    specs = specifications(manifest)
    network = manifest["network"]
    locks = {s.client: threading.Lock() for s in specs.values()}
    slots = {s.client: [slot for slot in manifest["slots"] if slot["client"] == s.client] for s in specs.values()}
    cursors = {c: 0 for c in slots}
    counters = {c: Counter() for c in slots}
    events = []
    event_lock = threading.Lock()

    def record(event):
        with event_lock:
            events.append(event)

    def active(slot, now):
        contact = manifest["contacts"][slot["contact"]]
        return (slot["start"] <= now < slot["end"] and contact["start"] <= now < contact["end"]
                and now < manifest["deadline_seconds"] and not outage_at(slot, manifest["outages"]))

    def send(connection, data, client, direction, slot, event):
        for offset in range(0, len(data), network["segment_bytes"]):
            block = data[offset:offset + network["segment_bytes"]]
            if not active(slot, time.monotonic() - epoch.value):
                raise RadioClosed("Radio closed before segment")
            time.sleep(len(block) * 8 / network["rate_bps"])
            now = time.monotonic() - epoch.value
            if not active(slot, now):
                raise RadioClosed("Radio closed during paced segment")
            connection.sendall(block)
            after = time.monotonic() - epoch.value
            event["last_send"] = after
            event["first_send"] = event.get("first_send", now)
            event[direction + "_bytes"] = event.get(direction + "_bytes", 0) + len(block)
            counters[client][direction + "_bytes"] += len(block)
            if after >= slot["end"]:
                event["boundary_overrun"] = True
                raise RadioClosed("Socket write crossed slot boundary")

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            try:
                message, raw = read_header(self.request)
                spec = specs[message["transfer_id"]]
                client = spec.client
                with locks[client]:
                    counters[client]["connections"] += 1
                    while not stop.is_set():
                        if message["transfer_id"] != spec.transfer_id:
                            raise ValueError("Connection changed client identity")
                        body = receive_exact(self.request, message["body_bytes"])
                        now = time.monotonic() - epoch.value
                        while cursors[client] < len(slots[client]) and slots[client][cursors[client]]["start"] < now + manifest["admission_lead_seconds"]:
                            cursors[client] += 1
                        if cursors[client] == len(slots[client]):
                            time.sleep(max(0, epoch.value + manifest["deadline_seconds"] - time.monotonic()))
                            return
                        slot = slots[client][cursors[client]]
                        cursors[client] += 1
                        time.sleep(max(0, epoch.value + slot["start"] - time.monotonic()))
                        event = {"slot": slot["id"], "client": client, "station": slot["station"],
                                 "op": message["op"], "position": message.get("position"),
                                 "start": time.monotonic() - epoch.value, "outcome": "pending"}
                        if outage_at(slot, manifest["outages"]):
                            counters[client]["outage_slots"] += 1
                            event["outcome"] = "outage"
                            record(event)
                            return
                        try:
                            if not active(slot, time.monotonic() - epoch.value):
                                raise RadioClosed("Slot expired before connect")
                            with connect(backend, frame_timeout(manifest)) as upstream:
                                event["link_open"] = time.monotonic() - epoch.value
                                time.sleep(network["rtt_seconds"] / 2 + slot["delay_seconds"])
                                send(upstream, raw, client, "request", slot, event)
                                send(upstream, body, client, "request", slot, event)
                                response, response_raw = read_header(upstream)
                                response_body = receive_exact(upstream, response["body_bytes"])
                                time.sleep(network["rtt_seconds"] / 2 + slot["delay_seconds"])
                                send(self.request, response_raw + response_body, client, "response", slot, event)
                            event["link_close"] = time.monotonic() - epoch.value
                            event["outcome"] = "complete"
                        except RadioClosed as exc:
                            event["outcome"] = "slot_overrun"
                            event["detail"] = str(exc)
                            record(event)
                            return
                        record(event)
                        message, raw = read_header(self.request)
            except (EOFError, OSError):
                pass
            except Exception as exc:
                record({"error": type(exc).__name__, "detail": str(exc)})

    with Server(("127.0.0.1", 0), Handler) as server:
        server.frame_timeout = frame_timeout(manifest)
        server.timeout = 0.1
        ready.put({"role": "proxy", "address": server.server_address, "pid": os.getpid()})
        go.wait()
        while not stop.is_set():
            server.handle_request()
    save(output / "proxy.json", {"pid": os.getpid(), "events": sorted(events, key=lambda e: e.get("start", 0)),
                                 "clients": {str(k): dict(v) for k, v in counters.items()}})


def audit_slots(manifest):
    checks = {}
    for slot in manifest["slots"]:
        contact = manifest["contacts"][slot["contact"]]
        checks[f"slot-{slot['id']}:contact"] = (contact["client"] == slot["client"] and contact["station"] == slot["station"]
            and contact["start"] <= slot["start"] < slot["end"] <= min(contact["end"], manifest["deadline_seconds"]))
    for key in ("station", "client"):
        for identity in sorted({s[key] for s in manifest["slots"]}, key=str):
            intervals = sorted((s["start"], s["end"]) for s in manifest["slots"] if s[key] == identity)
            checks[f"half_duplex:{key}:{identity}"] = all(left[1] <= right[0] + 1e-9 for left, right in zip(intervals, intervals[1:]))
    return checks


def run_manifest(manifest, source, output):
    predicted = predict_contacts(manifest)
    if not all(audit_slots(manifest).values()):
        raise ValueError("Invalid station/client/contact reservations")
    output = Path(output)
    execute_case(output, manifest, source, proxy_target=contact_proxy)
    save(output / "prediction.json", predicted)
    return summarize_case(output)


def summarize_case(output):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    prediction = json.loads((output / "prediction.json").read_text())
    coordinator = json.loads((output / "coordinator.json").read_text())
    proxy = json.loads((output / "proxy.json").read_text())
    specs = {s.client: s for s in specifications(manifest).values()}
    checks = audit_slots(manifest)
    checks["services_no_errors"] = not any("error" in e for e in coordinator["events"] + proxy["events"])
    checks["no_slot_overruns"] = all(e.get("outcome") in ("complete", "outage") for e in proxy["events"])
    checks["seven_distinct_processes"] = len({p["pid"] for p in json.loads((output / "processes.json").read_text())}) == 7
    rows = []
    for client, spec in specs.items():
        actual = json.loads((output / f"client-{client}.json").read_text())
        expected = prediction["clients"][str(client)]
        counts = proxy["clients"][str(client)]
        events = [e for e in proxy["events"] if e.get("client") == client]
        checks[f"client-{client}:delivery_prediction"] = actual["delivered"] == expected["delivered"]
        checks[f"client-{client}:request_bytes"] = counts.get("request_bytes", 0) == expected["request_bytes"]
        checks[f"client-{client}:response_bytes"] = counts.get("response_bytes", 0) == expected["response_bytes"]
        checks[f"client-{client}:outage_count"] = counts.get("outage_slots", 0) == expected["failed_slots"]
        checks[f"client-{client}:slot_sequence"] = [e["slot"] for e in events] == [e["slot"] for e in prediction["trace"] if e["client"] == client]
        eligible = [e for e in coordinator["events"] if e.get("client") == client and e.get("response", {}).get("new") is True]
        checks[f"client-{client}:unique_eligibility"] = len(eligible) == int(expected["eligible"])
        for event in events:
            slot = manifest["slots"][event["slot"]]
            if "first_send" in event:
                checks[f"event-{event['slot']}:contact_boundary"] = slot["start"] <= event["first_send"] <= event["last_send"] < slot["end"]
                checks[f"event-{event['slot']}:socket_inside_contact"] = slot["start"] <= event["link_open"] < slot["end"] and event.get("link_close", slot["end"]) <= slot["end"]
        path = output / "received" / f"client-{client}.safetensors"
        error = None
        if eligible:
            checks[f"client-{client}:export_hash"] = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == spec.payload_sha256
        else:
            checks[f"client-{client}:no_export"] = not path.exists()
        if actual["delivered"]:
            checks[f"client-{client}:deadline"] = actual["completed_seconds"] < manifest["deadline_seconds"]
            if expected["completed_seconds"] is not None:
                error = actual["completed_seconds"] - expected["completed_seconds"]
                checks[f"client-{client}:completion_error"] = abs(error) <= manifest["timing_tolerance_seconds"]
        rows.append({"client": client, "predicted_delivered": expected["delivered"], "measured_delivered": actual["delivered"],
                     "predicted_eligible": expected["eligible"], "measured_eligible": bool(eligible),
                     "predicted_seconds": expected["completed_seconds"], "measured_seconds": actual.get("completed_seconds"),
                     "completion_error_seconds": error, "predicted_request_bytes": expected["request_bytes"],
                     "measured_request_bytes": counts.get("request_bytes", 0), "predicted_response_bytes": expected["response_bytes"],
                     "measured_response_bytes": counts.get("response_bytes", 0), "outage_slots": counts.get("outage_slots", 0)})
    checks["eligibility_count"] = coordinator["eligible_count"] == sum(r["measured_eligible"] for r in rows)
    checks["storage_bound"] = coordinator["persisted_payload_bytes"] <= manifest["queue_bytes"]
    radio_events = [e for e in proxy["events"] if "link_open" in e and "link_close" in e]
    for key in ("station", "client"):
        for identity in {e[key] for e in radio_events}:
            intervals = sorted((e["link_open"], e["link_close"]) for e in radio_events if e[key] == identity)
            checks[f"measured_exclusive:{key}:{identity}"] = all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
    result = {"status": "passed" if all(checks.values()) else "failed", "case": manifest["case"],
              "checks": checks, "rows": rows, "contention_slots": sum(len(s["competing_clients"]) > 1 for s in manifest["slots"]),
              "executed_transactions": len(radio_events), "failed_checks": [k for k, v in checks.items() if not v],
              "scope": "Actual saved-adapter TCP replay under uniformly time-compressed nominal visibility, fixed TDMA station/client reservations and slot-aligned exogenous outages. No new training or physical spacecraft."}
    save(output / "results.json", result)
    return result


def build_manifest(config, case, source_data, raw_contacts):
    replay = config["replay_seconds"]
    compression = config["mission_seconds"] / replay
    contacts = [{**c, "start": c["start"] / compression, "end": c["end"] / compression,
                 "delay_seconds": c["delay_seconds"] / compression} for c in raw_contacts]
    base = "opaque-archived-adapter@" + hashlib.sha256(source_data).hexdigest()
    transfers = [asdict(TransferSpec.from_bytes(f"client-{i}-v0", i, 0, base, source_data, config["chunk_bytes"])) for i in range(5)]
    slots = allocate_slots(contacts, replay, config["slot_seconds"])
    outages = make_outages(slots, case["fault_seed"], config["slot_seconds"])
    return {"case": case, "base_identity": base, "transfers": transfers, "contacts": contacts,
            "slots": slots, "outages": outages, "deadline_seconds": replay,
            "socket_idle_timeout_seconds": replay + 5, "condition": "orbital-contacts",
            "queue_bytes": config["queue_bytes"], "time_compression": compression,
            "nominal_mission_rate_bps": config["rate_bps"] / compression,
            "processing_guard_seconds": config["processing_guard_seconds"],
            "admission_lead_seconds": config["admission_lead_seconds"],
            "timing_tolerance_seconds": config["timing_tolerance_seconds"],
            "network": {k: config[k] for k in ("rate_bps", "segment_bytes", "rtt_seconds", "outage_seconds", "retry_seconds")}}


def run_case(config_path, index, output):
    config_path = project_path(config_path)
    config = json.loads(config_path.read_text())
    case = config["cases"][index]
    source = project_path(config["payload"])
    source_data = source.read_bytes()
    if len(source_data) * 5 > config["queue_bytes"]:
        raise ValueError("Storage reservation bound exceeded")
    contacts_path = project_path(case["contacts"])
    raw = json.loads(contacts_path.read_text())["contacts"]
    if case.get("co_visible_stress"):
        selected = [c for c in raw if c["client"] == 3]
        raw = [{**c, "client": client} for c in selected for client in range(5)]
    manifest = build_manifest(config, case, source_data, raw)
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "prediction-before-run.json", predict_contacts(manifest))
    save(output / "config.json", config)
    files = [Path(__file__).resolve(), Path(__file__).resolve().with_name("socket_transport.py"),
             Path(__file__).resolve().with_name("reliable_transfer.py"), config_path,
             ROOT / "scripts/orbital_contact_transport.py", ROOT / "scripts/orbital_contact_report.py",
             ROOT / "tests/orbital_sft/test_contact_transport.py", ROOT / "infra/rivanna/contact_transport.sbatch"]
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(ROOT)))
    save(output / "provenance.json", {"input_sha256": hashlib.sha256(source_data).hexdigest(),
         "contacts_sha256": hashlib.sha256(contacts_path.read_bytes()).hexdigest(),
         "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
         "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "case_index": index})
    result = run_manifest(manifest, source, output / "execution")
    save(output / "results.json", result)
    return result


def plan_study(config_path, output):
    config_path = project_path(config_path)
    config = json.loads(config_path.read_text())
    data = project_path(config["payload"]).read_bytes()
    output = project_path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, case in enumerate(config["cases"]):
        raw = json.loads(project_path(case["contacts"]).read_text())["contacts"]
        if case.get("co_visible_stress"):
            raw = [{**c, "client": client} for c in raw if c["client"] == 3 for client in range(5)]
        manifest = build_manifest(config, case, data, raw)
        prediction = predict_contacts(manifest)
        assert all(audit_slots(manifest).values())
        save(output / f"manifest-{index}.json", manifest)
        save(output / f"prediction-{index}.json", prediction)
        rows.append({"case": index, "name": case["name"], "slots": len(manifest["slots"]),
                     "outages": len(manifest["outages"]),
                     "predicted_eligible": sum(c["eligible"] for c in prediction["clients"].values()),
                     "predicted_confirmed": sum(c["delivered"] for c in prediction["clients"].values())})
    save(output / "summary.json", {"status": "planned", "cases": rows,
         "scope": "Pre-execution protocol predictions, not measured transfers or learning quality"})
    return rows
