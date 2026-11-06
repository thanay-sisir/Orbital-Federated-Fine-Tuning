import csv
import fcntl
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path

from .aggregation import aggregate
from .checkpoint import atomic_json, read
from .config import digest, project_path
from .contacts import generate
from .events import EventQueue
from .mission_runner import apply_progress, bind_receipt, compatibility
from .network import LinkEngine
from .recovery_state import FaultInjector, MissionLedger, atomic_bytes
from .reliable_transfer import DurableReceiver, TransferSpec
from .round_planner import plan_round
from .runner import environment_identity, make_run, pool, require_allocation, source_identity


def task_contract(cfg, state, task, round_id):
    return {"config_hash": cfg.fingerprint(), "global_sha256": state["global_sha256"],
            "client": task["client"], "cursor": state["cursors"][task["client"]],
            "round": round_id, "version": state["version"], "local_steps": task["steps"]}


def deliver(receiver, receipt, payload, version, base, fault, round_id, output):
    bind_receipt(receipt, payload)
    spec = TransferSpec.from_bytes(receipt["id"], receipt["client"], version, base,
                                   payload.read_bytes(), receiver.chunk_bytes)
    receiver.begin(spec)
    missing = set(receiver.missing(spec.transfer_id))
    restored = len(spec.chunk_sha256) - len(missing)
    with payload.open("rb") as stream:
        for index, chunk in enumerate(receipt["chunks"]):
            if index not in missing:
                continue
            stream.seek(chunk["offset"])
            receiver.put(spec.transfer_id, index, stream.read(chunk["payload_bytes"]))
            fault.hit("after_chunk", round=round_id, client=spec.client, chunk=index, direction=receipt["direction"])
    receipt["durable_chunks_reused"] = restored
    receipt["durable_verified_bytes"] = sum(min(spec.chunk_bytes, spec.size - i * spec.chunk_bytes) for i in range(len(spec.chunk_sha256)) if i not in receiver.missing(spec.transfer_id))
    if receipt["complete"]:
        receiver.finalize(spec.transfer_id)
        fault.hit("after_eligibility", round=round_id, client=spec.client, direction=receipt["direction"])
        receiver.export(spec.transfer_id, output)
        if digest(output) != spec.payload_sha256:
            raise ValueError("Receiver export does not match planned payload")
        receipt["receiver_export"] = str(output)


def materialize(path, ledger):
    rounds = ledger.rounds()
    for filename, field in (("transfers.jsonl", "receipts"), ("events.jsonl", "events"), ("client_updates.jsonl", "results")):
        rows = [row for entry in rounds for row in entry[field]]
        atomic_bytes(path / filename, "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows).encode())
    fields = ["round", "mission_seconds", "version", "participants", "performed_steps", "accepted_steps", "wire_bytes", "validation_answer_loss", "attempt", "attempt_elapsed_seconds"]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for entry in rounds:
        writer.writerow(entry["metric"])
    atomic_bytes(path / "metrics.csv", stream.getvalue().encode())
    state, adapter = ledger.latest()
    model_path = project_path(state["global_adapter"])
    if not model_path.exists() or digest(model_path) != state["global_sha256"]:
        atomic_bytes(model_path, adapter)
    atomic_json(path / "checkpoint.json", state)
    atomic_json(path / "work_accounting.json", {k: v for k, v in state.items() if k not in {"link", "source_sha256"}})
    return state


def write_outage_execution(path, link, mission_seconds):
    from .outages import audit

    atomic_json(path / "outage_execution.json", {"audit": audit(link, mission_seconds),
        "trace_sha256": link.trace_sha256, "mission_clock": link.mission_clock,
        "observations": link.observations, "decisions": link.decisions,
        "attempts": link.attempts, "compute_events": link.compute_events,
        "scope": "Exogenous outages consume mission time; policies receive nominal contacts and observed availability only. Partial optimizer-step compute cost is modeled, not GPU-executed."})


def run_resilient_mission(cfg, run_id, resume=False, fault_plan=None, fault_mode="exit", outage_trace=None):
    from safetensors.torch import load_file, save_file
    from swarmshield.posttraining.model_loader import fixed_a_hash
    from swarmshield.posttraining.trainer import assess, execute, initialize
    from .outages import OutageEngine, audit, plan_outage_round

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_id):
        raise ValueError("Invalid run identity")
    if cfg.training.full_shard_pass or cfg.training.epoch_number != 1 or cfg.training.parent_run:
        raise ValueError("Recoverable mission starts from the base with partial participation")
    if cfg.training.workers != cfg.training.clients or cfg.training.evaluate_test:
        raise ValueError("Recoverable mission requires one worker per client and validation-only evaluation")
    if cfg.network.chunk_bytes > 4 * 1024 * 1024:
        raise ValueError("Durable receiver supports chunks up to four MiB")
    plan = read(project_path(fault_plan)) if fault_plan else []
    if not isinstance(plan, list):
        raise ValueError("Fault plan must be a list")
    trace = read(project_path(outage_trace)) if outage_trace else None
    if trace is None and cfg.network.policy == "retry_backoff":
        raise ValueError("Retry/backoff requires an outage trace")
    if trace is not None and cfg.network.policy not in {"fixed", "retry_backoff", "feasible_fair"}:
        raise ValueError("Unsupported policy for outage execution")
    require_allocation(cfg)
    compatibility(cfg)
    identity = {"config_hash": cfg.fingerprint(), "source_sha256": source_identity(),
                "environment_identity": environment_identity(),
                "prepared_manifest_sha256": digest(project_path(cfg.data.prepared_path) / "manifest.json"),
                "fault_plan": plan, "outage_trace": trace, "schema_version": 1}
    if resume:
        path = project_path("results/orbital") / run_id
        manifest = read(path / "manifest.json")
        if manifest.get("recovery_contract") != identity or not (path / "mission.sqlite").exists():
            raise ValueError("Resume requires an existing resilient run with matching identities")
    else:
        path = make_run(cfg, run_id, "resilient_mission_training")
        manifest = read(path / "manifest.json")
        manifest["recovery_contract"] = identity
        atomic_json(path / "manifest.json", manifest)
    lock = (path / "coordinator.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("Another coordinator owns this run")
    ledger = None
    workers = []
    started = time.perf_counter()
    attempt = None
    try:
        ledger = MissionLedger(path / "mission.sqlite", identity)
        fault = FaultInjector(path / "faults_consumed.json", plan, fault_mode)
        attempt = ledger.attempt()
        attempt_path = path / f"attempt-{attempt:04d}"
        attempt_path.mkdir()
        finished = ledger.finished()
        if finished:
            for name, expected in finished["export_sha256"].items():
                if digest(project_path(finished["export_path"]) / name) != expected:
                    raise ValueError("Completed PEFT export checksum mismatch")
            state = materialize(path, ledger)
            if trace is not None:
                restored_link = OutageEngine(read(path / "effective_contacts.json"), cfg.network, trace, cfg.training.clients, cfg.orbit.mission_seconds)
                restored_link.restore(state["link"])
                write_outage_execution(path, restored_link, cfg.orbit.mission_seconds)
            alias = path / "export"
            if not alias.exists():
                alias.symlink_to(project_path(finished["export_path"]).relative_to(path), target_is_directory=True)
            atomic_json(path / "evaluation.json", finished["evaluation"])
            atomic_json(path / "health.json", {"status": "completed", "global_version": state["version"], "attempts": attempt, "resume_scope": "idempotent completed-run inspection"})
            ledger.attempt_done(attempt, "completed_no_training")
            return path
        latest = ledger.latest()
        n = cfg.training.clients
        if latest:
            state, blob = latest
            global_path = project_path(state["global_adapter"])
            atomic_bytes(global_path, blob)
        else:
            state = {"data_seed": cfg.seed, "training_seed": cfg.training_seed, "next_round": 0,
                     "cursors": [0] * n, "age": [0] * n, "version": 0, "performed_steps": 0,
                     "delivered_steps": 0, "accepted_steps": 0, "performed_examples": 0,
                     "accepted_examples": 0, "accepted_updates": [0] * n, "accepted_digests": [[] for _ in range(n)]}
            global_path = None
        workers = [pool() for _ in range(cfg.training.workers)]
        for device, worker in enumerate(workers):
            initialized = worker.submit(initialize, cfg.model_dump(), device, str(attempt_path / f"initial-{device}"),
                                        str(global_path) if global_path is not None else None).result()
            if global_path is None:
                global_path = Path(initialized)
            if digest(initialized) != digest(global_path):
                raise ValueError("Restored worker adapter differs from committed global state")
        if latest is None:
            geometry = read(generate(cfg))
            atomic_json(path / "contacts.json", geometry)
            contacts = geometry["contacts"]
            if cfg.network.policy == "connected":
                contacts = [{"client": i, "station": "connected-ground", "start": 0, "end": cfg.orbit.mission_seconds, "delay_seconds": 0} for i in range(n)]
            atomic_json(path / "effective_contacts.json", contacts)
            link = OutageEngine(contacts, cfg.network, trace, n, cfg.orbit.mission_seconds) if trace is not None else LinkEngine(contacts, cfg.network)
            committed = path / "initial_committed.safetensors"
            atomic_bytes(committed, global_path.read_bytes())
            global_path = committed
            state.update({"global_adapter": str(global_path), "global_sha256": digest(global_path),
                          "link": link.snapshot(), "config_hash": cfg.fingerprint(), "source_sha256": identity["source_sha256"]})
            ledger.commit(-1, state, global_path.read_bytes(), {})
        else:
            contacts = read(path / "effective_contacts.json")
            link = OutageEngine(contacts, cfg.network, trace, n, cfg.orbit.mission_seconds) if trace is not None else LinkEngine(contacts, cfg.network)
            link.restore(state["link"])
        if not (path / "base_validation.json").exists():
            atomic_json(path / "base_validation.json", workers[0].submit(assess, cfg.model_dump(), 0,
                        str(path / "initial_committed.safetensors"), "", False, True).result())
        materialize(path, ledger)
        base = cfg.model.identity + "@" + cfg.model.revision
        for round_id in range(state["next_round"], cfg.training.rounds):
            start = round_id * cfg.network.round_seconds
            deadline = min(start + cfg.network.round_seconds, cfg.orbit.mission_seconds)
            if start >= deadline:
                break
            round_path = path / f"round-{round_id:04d}"
            round_path.mkdir(exist_ok=True)
            global_state = load_file(str(global_path))
            down_path = round_path / "global_B.safetensors"
            save_file({k: v for k, v in global_state.items() if "lora_B" in k}, str(down_path))
            old_ids = set(link.identities)
            planner = plan_outage_round if trace is not None else plan_round
            link, tasks, predicted = planner(link, n, start, deadline, down_path.stat().st_size,
                                               cfg.training.device_step_seconds, cfg.training.local_steps,
                                               cfg.network.policy, state["age"], round_id)
            receipts = [r for r in link.receipts if r["id"] not in old_ids]
            receipt_map = {r["id"]: r for r in receipts}
            atomic_json(round_path / "plan.json", {"tasks": tasks, "predicted": predicted})
            down_receiver = DurableReceiver(round_path / "downloads.sqlite", base, cfg.network.queue_bytes)
            up_receiver = DurableReceiver(round_path / "uploads.sqlite", base, cfg.network.queue_bytes)
            for receiver in (down_receiver, up_receiver):
                receiver.chunk_bytes = cfg.network.chunk_bytes
                receiver.advance(state["version"])
            try:
                for receipt in receipts:
                    if receipt["direction"] == "download":
                        deliver(down_receiver, receipt, down_path, state["version"], base, fault, round_id,
                                round_path / f"received-{receipt['id']}.safetensors")
                pending, results, delivered = [], [], []
                for device, task in enumerate(tasks):
                    if not receipt_map[task["download_id"]]["complete"]:
                        raise ValueError("Training requires complete durable model delivery")
                    contract = task_contract(cfg, state, task, round_id)
                    cache_path = round_path / f"client-{task['client']}-cache.json"
                    if cache_path.exists():
                        cached = read(cache_path)
                        if cached["contract"] != contract or digest(project_path(cached["result"]["upload_path"])) != cached["result"]["upload_sha256"]:
                            raise ValueError("Cached local update identity or checksum mismatch")
                        pending.append((task, cached["result"], None))
                    else:
                        output = attempt_path / f"r{round_id}-client-{task['client']}"
                        payload = {"config": cfg.model_dump(), "device": device, "output": str(output),
                                   "client": task["client"], "cursor": contract["cursor"], "round": round_id,
                                   "version": state["version"], "global_adapter": str(global_path), "local_steps": task["steps"]}
                        pending.append((task, None, workers[device].submit(execute, payload)))
                for task, cached, future in pending:
                    result = cached if cached is not None else future.result()
                    client = task["client"]
                    if result["client"] != client or result["version"] != state["version"] or result["base_identity"] != base:
                        raise ValueError("Local update identity or model version mismatch")
                    if result["steps"] != task["steps"] or result["cursor"] - state["cursors"][client] != task["steps"] * cfg.training.accumulation:
                        raise ValueError("Executed work differs from committed cursor and mission plan")
                    payload_path = project_path(result["upload_path"])
                    if digest(payload_path) != result["upload_sha256"]:
                        raise ValueError("Local update payload checksum mismatch")
                    atomic_json(round_path / f"client-{client}-cache.json", {"contract": task_contract(cfg, state, task, round_id), "result": result})
                    receipt = receipt_map[task["upload_id"]]
                    received = round_path / f"received-{receipt['id']}.safetensors"
                    deliver(up_receiver, receipt, payload_path, state["version"], base, fault, round_id, received)
                    result = {**result, "receipt": receipt}
                    results.append(result)
                    if receipt["complete"]:
                        delivered.append({**result, "tensors": load_file(str(received))})
                merged, membership, rejected = aggregate(delivered, set(range(n)), state["version"], base,
                    cfg.network.min_participants, expected_shapes={k: tuple(v.shape) for k, v in global_state.items() if "lora_B" in k},
                    expected_a_hash=fixed_a_hash(global_state), expected_contract=cfg.model.model_dump())
                if set(membership) != set(predicted["accepted_clients"]) or bool(merged is not None) != predicted["successful"]:
                    raise ValueError("Durable aggregation differs from predicted delivery")
                next_state = json.loads(json.dumps(state))
                apply_progress(next_state, results, membership)
                if trace is not None:
                    audit(link, deadline)
                    next_state["failed_transfer_attempts"] = next_state.get("failed_transfer_attempts", 0) + predicted["failed_attempts"]
                    next_state["wasted_simulated_compute_seconds"] = next_state.get("wasted_simulated_compute_seconds", 0) + predicted["wasted_compute_seconds"]
                    next_state["mission_clock"] = deadline
                if merged is not None:
                    global_state.update(merged)
                    candidate = round_path / "shared_adapter.safetensors"
                    save_file(global_state, str(candidate))
                    next_state["version"] += 1
                else:
                    candidate = global_path
                evaluation = workers[0].submit(assess, cfg.model_dump(), 0, str(candidate), "", False).result()
                queue = EventQueue()
                queue.clock = start
                for task in tasks:
                    queue.push(task["ready"], "model_delivered", {"client": task["client"]})
                    queue.push(task["complete"], "training_complete", {"client": task["client"], "steps": task["steps"]})
                for receipt in receipts:
                    for chunk in receipt["chunks"]:
                        queue.push(chunk["arrival"], "chunk_verified", {"id": receipt["id"], **chunk})
                if trace is not None:
                    for observation in predicted["outage_observations"]:
                        payload = {k: v for k, v in observation.items() if k != "kind"}
                        payload["outage_kind"] = observation["kind"]
                        queue.push(observation["time"], observation["event"], payload)
                    for event in predicted["compute_events"]:
                        if not event["completed"]:
                            queue.push(event["end"], "incomplete_optimizer_step", event)
                queue.push(deadline, "round_closed", {"participants": membership, "rejected": rejected, "version": next_state["version"]}, 10)
                events = [{"time": e.time, "kind": e.kind, **e.payload} for e in queue.drain()]
                next_state.update({"next_round": round_id + 1, "global_adapter": str(candidate), "global_sha256": digest(candidate), "link": link.snapshot()})
                metric = {"round": round_id, "mission_seconds": deadline, "version": next_state["version"], "participants": len(membership),
                          "performed_steps": next_state["performed_steps"], "accepted_steps": next_state["accepted_steps"], "wire_bytes": link.wire_bytes,
                          "validation_answer_loss": evaluation["validation"]["answer_loss"], "attempt": attempt, "attempt_elapsed_seconds": time.perf_counter() - started}
                fault.hit("before_round_commit", round=round_id)
                ledger.commit(round_id, next_state, candidate.read_bytes(), {"metric": metric, "receipts": receipts, "results": results, "events": events})
                fault.hit("after_round_commit", round=round_id)
                state = materialize(path, ledger)
                global_path = project_path(state["global_adapter"])
                atomic_json(round_path / "checkpoint.json", state)
            finally:
                down_receiver.close()
                up_receiver.close()
        export = attempt_path / "export"
        final = workers[0].submit(assess, cfg.model_dump(), 0, str(global_path), str(export), True, True).result()
        export_sha = {str(p.relative_to(export)): digest(p) for p in export.rglob("*") if p.is_file()}
        ledger.finish({"evaluation": final, "export_path": str(export), "export_sha256": export_sha})
        link_path = path / "export"
        if not link_path.exists():
            link_path.symlink_to(export.relative_to(path), target_is_directory=True)
        atomic_json(path / "evaluation.json", final)
        if trace is not None:
            write_outage_execution(path, link, cfg.orbit.mission_seconds)
        atomic_json(path / "health.json", {"status": "completed", "global_version": state["version"], "attempts": attempt,
                    "scope": "Real local training and atomic round-state commits; exogenous outages consume mission time when configured; coordinator process recovery replays uncommitted modeled rounds"})
        ledger.attempt_done(attempt, "completed")
        return path
    except BaseException as error:
        if ledger is not None and attempt is not None:
            ledger.attempt_done(attempt, "interrupted")
        atomic_json(path / "health.json", {"status": "interrupted", "reason": str(error), "resume_required": True})
        raise
    finally:
        for worker in workers:
            worker.shutdown(wait=True, cancel_futures=True)
        if ledger is not None:
            ledger.close()
        lock.close()
