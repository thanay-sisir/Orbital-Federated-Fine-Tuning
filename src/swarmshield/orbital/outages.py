import copy
import hashlib
import heapq
import json
import math
from dataclasses import dataclass

from .network import LinkEngine


@dataclass(frozen=True)
class PolicyView:
    clock: float
    deadline: float
    client: int
    age: int
    payload_bytes: int
    step_seconds: float
    max_steps: int
    contacts: tuple
    reservations: tuple
    rate_bps: float
    chunk_bytes: int
    framing_bytes: int
    ack_bytes: int
    acquisition_seconds: float


def admit_steps(view, policy):
    if policy not in {"fixed", "retry_backoff", "feasible_fair"}:
        raise ValueError("Unsupported outage policy")
    if policy != "feasible_fair":
        return view.max_steps
    from types import SimpleNamespace

    settings = SimpleNamespace(rate_bps=view.rate_bps, chunk_bytes=view.chunk_bytes,
        framing_bytes=view.framing_bytes, ack_bytes=view.ack_bytes,
        acquisition_seconds=view.acquisition_seconds, byte_budget=2**63 - 1,
        queue_bytes=2**63 - 1)
    contacts = [dict(zip(("client", "station", "start", "end", "delay_seconds"), c)) for c in view.contacts]
    for steps in sorted({1, max(1, view.max_steps // 2), view.max_steps}, reverse=True):
        forecast = LinkEngine(contacts, settings)
        for station, client, start, end in view.reservations:
            forecast.busy.setdefault(station, []).append([start, end])
            forecast.client_busy.setdefault(client, []).append([start, end])
        down = forecast.transfer(view.client, view.payload_bytes, view.clock, view.deadline, "download", "forecast-down")
        if down["complete"]:
            up = forecast.transfer(view.client, view.payload_bytes, down["arrival"] + steps * view.step_seconds,
                                   view.deadline, "upload", "forecast-up")
            if up["complete"]:
                return steps
    return 0


def validate_trace(trace, clients, stations, mission_seconds):
    if set(trace) != {"schema_version", "seed", "profile", "events"} or trace["schema_version"] != 1:
        raise ValueError("Invalid outage trace schema")
    for event in trace["events"]:
        if event.get("kind") not in {"link", "client", "station"}:
            raise ValueError("Invalid outage kind")
        expected = {"kind", "start", "end", "station" if event["kind"] == "station" else "client"}
        if set(event) != expected or not all(type(event[k]) in (int, float) and math.isfinite(event[k]) for k in ("start", "end")):
            raise ValueError("Invalid outage fields")
        if not 0 <= event["start"] < event["end"] <= mission_seconds:
            raise ValueError("Outage outside mission")
        if event["kind"] == "station":
            if event["station"] not in stations:
                raise ValueError("Unknown outage station")
        elif type(event["client"]) is not int or not 0 <= event["client"] < clients:
            raise ValueError("Unknown outage client")


class OutageEngine(LinkEngine):
    def __init__(self, contacts, settings, trace, clients, mission_seconds):
        super().__init__(copy.deepcopy(contacts), settings)
        validate_trace(trace, clients, {c["station"] for c in contacts}, mission_seconds)
        self._trace = copy.deepcopy(trace)
        self.trace_sha256 = hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()
        self.mission_clock = 0.0
        self.observations = []
        self.decisions = []
        self.attempts = []
        self.compute_events = []

    def snapshot(self):
        return {**super().snapshot(), "trace_sha256": self.trace_sha256,
                "mission_clock": self.mission_clock, "observations": self.observations,
                "decisions": self.decisions, "attempts": self.attempts, "compute_events": self.compute_events}

    def restore(self, state):
        if state["trace_sha256"] != self.trace_sha256:
            raise ValueError("Outage trace changed across resume")
        super().restore(state)
        self.mission_clock = state["mission_clock"]
        self.observations = state["observations"]
        self.decisions = state["decisions"]
        self.attempts = state["attempts"]
        self.compute_events = state["compute_events"]


def plan_outage_round(engine, clients, start, deadline, payload_bytes, step_seconds, max_steps, policy, age, round_id):
    if policy not in {"fixed", "retry_backoff", "feasible_fair"} or start < engine.mission_clock or deadline <= start:
        raise ValueError("Invalid outage round")
    if payload_bytes <= 0 or clients * payload_bytes > engine.settings.queue_bytes or min(step_seconds, max_steps) <= 0:
        raise ValueError("Invalid payload or work budget")
    settings = engine.settings
    heap, sequence, clock = [], 0, float(start)
    blocked = {"station": {}, "link": {}, "client": {}}
    radios, computing = {}, {}
    chosen, done, last_compute = {}, [0] * clients, {}
    requests, receipts, observations = {}, {}, []
    all_attempts_start = len(engine.attempts)
    all_compute_start = len(engine.compute_events)
    wasted_compute = 0.0

    def push(time, priority, kind, value):
        nonlocal sequence
        if time <= deadline:
            heapq.heappush(heap, (float(time), priority, sequence, kind, value))
            sequence += 1

    def new_receipt(client, direction, ready):
        identity = f"r{round_id}-c{client}-{'down' if direction == 'download' else 'up'}"
        if identity in engine.identities:
            raise ValueError("Duplicate round transfer identity")
        engine.identities.add(identity)
        receipt = {"id": identity, "client": client, "direction": direction, "size": payload_bytes,
                   "verified_bytes": 0, "complete": False, "ready": ready, "arrival": None, "chunks": [],
                   "attempted_wire_bytes": 0, "failed_attempts": 0}
        receipts[identity] = receipt
        requests[client] = {"receipt": receipt, "retry_at": ready}
        return receipt

    for client in range(clients):
        new_receipt(client, "download", start)
    for index, event in enumerate(engine._trace["events"]):
        if event["start"] < start < event["end"]:
            key = event.get("station", event.get("client"))
            blocked[event["kind"]][key] = blocked[event["kind"]].get(key, 0) + 1
        if start <= event["start"] < deadline:
            push(event["start"], 1, "outage_start", (index, event))
        if start < event["end"] <= deadline:
            push(event["end"], 0, "outage_end", (index, event))
    for contact in engine.contacts:
        if start < contact["start"] <= deadline:
            push(contact["start"], 3, "wake", None)
    push(deadline, 4, "deadline", None)

    def active_client(client):
        return blocked["client"].get(client, 0) == 0

    def release_attempt(attempt, success, failure=None):
        station, client = attempt["station"], attempt["client"]
        receipt = receipts[attempt["id"]]
        actual_end = clock
        wire = attempt["planned_wire_bytes"] if success else min(attempt["planned_wire_bytes"],
            max(0, math.floor((actual_end - attempt["start"] - settings.acquisition_seconds) * settings.rate_bps / 8 + 1e-7)))
        reservation = attempt.pop("reservation")
        reservation[1] = actual_end
        client_reservation = attempt.pop("client_reservation")
        client_reservation[1] = actual_end
        engine.wire_bytes += wire
        receipt["attempted_wire_bytes"] += wire
        attempt.update({"finish": actual_end, "wire_bytes": wire, "success": success, "failure": failure})
        engine.attempts.append({k: v for k, v in attempt.items() if k != "token"})
        radios.pop(station, None)
        if success:
            chunk = {"station": station, "start": attempt["start"], "finish": actual_end,
                     "arrival": actual_end, "payload_bytes": attempt["payload_bytes"],
                     "wire_bytes": wire, "offset": receipt["verified_bytes"], "sha256": None}
            receipt["chunks"].append(chunk)
            receipt["verified_bytes"] += chunk["payload_bytes"]
            if receipt["verified_bytes"] == payload_bytes:
                receipt["complete"], receipt["arrival"] = True, clock
                requests.pop(client, None)
        else:
            receipt["failed_attempts"] += 1
            if client in requests:
                requests[client]["retry_at"] = clock + (30 if policy == "retry_backoff" else 0)
                if requests[client]["retry_at"] > clock:
                    push(requests[client]["retry_at"], 3, "wake", None)

    def begin_compute(client):
        if not active_client(client) or client in computing or done[client] >= chosen[client]:
            return
        token = (client, done[client], sequence)
        finish = clock + step_seconds
        computing[client] = {"token": token, "start": clock, "finish": finish}
        push(finish, 0, "step_end", (client, token))

    def schedule():
        for client in chosen:
            if client not in requests:
                begin_compute(client)
        while True:
            candidates = []
            occupied = {a["client"] for a in radios.values()}
            for client, request in requests.items():
                receipt = request["receipt"]
                if client in occupied or not active_client(client) or blocked["link"].get(client, 0) or request["retry_at"] > clock:
                    continue
                payload = min(payload_bytes - receipt["verified_bytes"], settings.chunk_bytes)
                wire = payload + settings.framing_bytes + settings.ack_bytes
                if engine.wire_bytes + sum(a["planned_wire_bytes"] for a in radios.values()) + wire > settings.byte_budget:
                    continue
                for contact in engine.contacts:
                    station = contact["station"]
                    if contact["client"] != client or station in radios or blocked["station"].get(station, 0):
                        continue
                    duration = settings.acquisition_seconds + wire * 8 / settings.rate_bps + contact["delay_seconds"]
                    if not contact["start"] <= clock < contact["end"] or clock + duration > min(contact["end"], deadline):
                        continue
                    steps = chosen.get(client)
                    if receipt["direction"] == "download" and steps is None:
                        view = PolicyView(clock, deadline, client, age[client], payload_bytes, step_seconds, max_steps,
                            tuple((c["client"], c["station"], c["start"], c["end"], c["delay_seconds"]) for c in engine.contacts),
                            tuple((s, a["client"], a["start"], a["nominal_finish"]) for s, a in radios.items()),
                            settings.rate_bps, settings.chunk_bytes, settings.framing_bytes, settings.ack_bytes, settings.acquisition_seconds)
                        steps = admit_steps(view, policy)
                        if not steps:
                            continue
                    priority = (0 if receipt["direction"] == "upload" else 1, -age[client], contact["end"], client) if policy == "feasible_fair" else (0 if receipt["direction"] == "download" else 1, client, contact["end"], client)
                    candidates.append((priority, station, client, steps, payload, wire, duration))
            if not candidates:
                return
            priority, station, client, steps, payload, wire, duration = min(candidates)
            receipt = requests[client]["receipt"]
            if client not in chosen:
                chosen[client] = steps
                engine.decisions.append({"time": clock, "round": round_id, "client": client, "steps": steps,
                                         "observations_seen": len(engine.observations), "policy": policy})
            token = (receipt["id"], receipt["verified_bytes"], sequence)
            reservation, client_reservation = [clock, clock + duration], [clock, clock + duration]
            engine.busy.setdefault(station, []).append(reservation)
            engine.client_busy.setdefault(client, []).append(client_reservation)
            attempt = {"id": receipt["id"], "client": client, "station": station, "start": clock,
                       "nominal_finish": clock + duration, "payload_bytes": payload, "planned_wire_bytes": wire,
                       "offset": receipt["verified_bytes"], "token": token,
                       "reservation": reservation, "client_reservation": client_reservation}
            radios[station] = attempt
            push(clock + duration, 0, "chunk_end", (station, token))

    push(start, 3, "wake", None)
    while heap:
        clock = heap[0][0]
        batch = []
        while heap and heap[0][0] == clock:
            batch.append(heapq.heappop(heap))
        for _, _, _, kind, value in batch:
            if kind == "chunk_end":
                station, token = value
                if station in radios and radios[station]["token"] == token:
                    release_attempt(radios[station], True)
            elif kind == "step_end":
                client, token = value
                if client in computing and computing[client]["token"] == token:
                    step = computing.pop(client)
                    done[client] += 1
                    last_compute[client] = clock
                    engine.compute_events.append({"round": round_id, "client": client, "start": step["start"], "end": clock, "completed": True})
                    if done[client] == chosen[client]:
                        new_receipt(client, "upload", clock)
            elif kind in {"outage_start", "outage_end"}:
                index, event = value
                key = event.get("station", event.get("client"))
                counters = blocked[event["kind"]]
                counters[key] = max(0, counters.get(key, 0) + (1 if kind == "outage_start" else -1))
                observed = {"time": clock, "round": round_id, "event": kind, "kind": event["kind"],
                            "target": key, "trace_event_index": index}
                engine.observations.append(observed)
                observations.append(observed)
                if kind == "outage_start":
                    for station, attempt in list(radios.items()):
                        affected = event["kind"] == "station" and station == key or event["kind"] != "station" and attempt["client"] == key
                        if affected and attempt["nominal_finish"] > clock:
                            release_attempt(attempt, False, event["kind"])
                    if event["kind"] == "client" and key in computing:
                        step = computing.pop(key)
                        wasted_compute += clock - step["start"]
                        engine.compute_events.append({"round": round_id, "client": key, "start": step["start"], "end": clock, "completed": False, "failure": "client"})
        if clock < deadline:
            schedule()
    clock = deadline
    for station, attempt in list(radios.items()):
        release_attempt(attempt, False, "deadline")
    for client, step in computing.items():
        wasted_compute += deadline - step["start"]
        engine.compute_events.append({"round": round_id, "client": client, "start": step["start"], "end": deadline, "completed": False, "failure": "deadline"})
    tasks = []
    for client in sorted(chosen):
        if not done[client]:
            continue
        up_id = f"r{round_id}-c{client}-up"
        if up_id not in receipts:
            new_receipt(client, "upload", deadline)
        tasks.append({"client": client, "steps": done[client], "admitted_steps": chosen[client],
                      "download_id": f"r{round_id}-c{client}-down", "upload_id": up_id,
                      "ready": receipts[f"r{round_id}-c{client}-down"]["arrival"], "complete": last_compute[client]})
    uploaded = [t for t in tasks if receipts[t["upload_id"]]["complete"]]
    successful = len(uploaded) >= settings.min_participants
    accepted = sorted(t["client"] for t in uploaded) if successful else []
    engine.receipts.extend(receipts.values())
    engine.mission_clock = deadline
    summary = {"round": round_id, "start": start, "deadline": deadline,
               "delivered_clients": [t["client"] for t in uploaded], "accepted_clients": accepted,
               "scheduled_steps": sum(t["steps"] for t in tasks),
               "accepted_steps": sum(t["steps"] for t in uploaded) if successful else 0,
               "successful": successful,
               "last_upload_arrival": max((receipts[t["upload_id"]]["arrival"] for t in uploaded), default=None),
               "failed_attempts": sum(not a["success"] for a in engine.attempts[all_attempts_start:]),
               "wasted_compute_seconds": wasted_compute,
               "outage_observations": observations,
               "compute_events": engine.compute_events[all_compute_start:]}
    return engine, tasks, summary


def audit(engine, mission_seconds):
    from collections import defaultdict

    occupied = defaultdict(list)
    total = 0
    for attempt in engine.attempts:
        if not 0 <= attempt["start"] <= attempt["finish"] <= mission_seconds:
            raise AssertionError("Attempt outside mission clock")
        if not 0 <= attempt["wire_bytes"] <= attempt["planned_wire_bytes"]:
            raise AssertionError("Invalid attempted wire accounting")
        capacity = max(0, attempt["finish"] - attempt["start"] - engine.settings.acquisition_seconds) * engine.settings.rate_bps / 8
        if attempt["wire_bytes"] > capacity + 1e-5:
            raise AssertionError("Attempt exceeds link capacity")
        total += attempt["wire_bytes"]
        occupied[("station", attempt["station"])].append((attempt["start"], attempt["finish"]))
        occupied[("client", attempt["client"])].append((attempt["start"], attempt["finish"]))
    for intervals in occupied.values():
        intervals.sort()
        if any(a[1] > b[0] + 1e-8 for a, b in zip(intervals, intervals[1:])):
            raise AssertionError("Half-duplex attempt overlap")
    if total != engine.wire_bytes or total > engine.settings.byte_budget:
        raise AssertionError("Byte budget or attempt accounting mismatch")
    for receipt in engine.receipts:
        offset = 0
        for chunk in receipt["chunks"]:
            if chunk["offset"] != offset or chunk["start"] < receipt["ready"] or chunk["arrival"] > mission_seconds:
                raise AssertionError("Noncausal or noncontiguous verified chunk")
            offset += chunk["payload_bytes"]
        if offset != receipt["verified_bytes"] or receipt["complete"] != (offset == receipt["size"]):
            raise AssertionError("Invalid delivery eligibility")
    return {"attempts": len(engine.attempts), "failed_attempts": sum(not a["success"] for a in engine.attempts), "wire_bytes": total}
