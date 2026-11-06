import csv
import hashlib
import time

from .aggregation import aggregate
from .checkpoint import append, atomic_json, read
from .config import digest, project_path
from .contacts import generate
from .events import EventQueue
from .network import LinkEngine
from .round_planner import plan_round
from .runner import environment_identity, make_run, pool, require_allocation, source_identity


def bind_receipt(receipt, payload):
    if payload.stat().st_size != receipt["size"]:
        raise ValueError("Actual serialized payload differs from planned byte budget")
    offset = 0
    with payload.open("rb") as stream:
        for chunk in receipt["chunks"]:
            if chunk["offset"] != offset:
                raise ValueError("Noncontiguous planned chunk offsets")
            block = stream.read(chunk["payload_bytes"])
            if len(block) != chunk["payload_bytes"]:
                raise ValueError("Actual payload is truncated")
            chunk["sha256"] = hashlib.sha256(block).hexdigest()
            offset += len(block)
    if offset != receipt["verified_bytes"] or (receipt["complete"] and offset != receipt["size"]):
        raise ValueError("Receipt byte verification mismatch")
    receipt["sha256"] = digest(payload)


def apply_progress(state, results, accepted):
    accepted = set(accepted)
    for result in results:
        client = result["client"]
        previous = state["cursors"][client]
        if result["cursor"] < previous:
            raise ValueError("Client cursor moved backwards")
        state["performed_examples"] += result["cursor"] - previous
        state["cursors"][client] = result["cursor"]
        state["performed_steps"] += result["steps"]
        if result["receipt"]["complete"]:
            state["delivered_steps"] += result["steps"]
        if client in accepted:
            state["accepted_steps"] += result["steps"]
            state["accepted_examples"] += result["cursor"] - previous
            state["accepted_updates"][client] += 1
            state["accepted_digests"][client] = sorted(set(state["accepted_digests"][client]) | set(result["example_digests"]))
    for client in range(len(state["cursors"])):
        state["age"][client] = 0 if client in accepted else state["age"][client] + 1


def compatibility(cfg):
    gate = read(project_path(cfg.training.compatibility_run) / "compatibility.json")
    if not gate["passed"] or gate["model"] != cfg.model.model_dump() or gate["max_length"] < cfg.data.max_length:
        raise ValueError("Matching current-source compatibility pilot required")
    if gate["environment_identity"] != environment_identity() or gate["source_sha256"] != source_identity():
        raise ValueError("Environment or source changed after compatibility pilot")


def run_mission(cfg, run_id):
    from safetensors.torch import load_file, save_file
    from swarmshield.posttraining.model_loader import fixed_a_hash
    from swarmshield.posttraining.trainer import assess, execute, initialize

    if cfg.network.policy == "retry_backoff":
        raise ValueError("Retry/backoff requires recoverable outage execution")

    if cfg.training.full_shard_pass or cfg.training.epoch_number != 1 or cfg.training.parent_run:
        raise ValueError("Mission comparison starts from the base with partial participation")
    if cfg.training.workers != cfg.training.clients or cfg.training.evaluate_test:
        raise ValueError("Mission comparison requires one worker per client and reserves test evaluation")
    require_allocation(cfg)
    compatibility(cfg)
    path = make_run(cfg, run_id, "mission_budget_training")
    started = time.perf_counter()
    workers = [pool() for _ in range(cfg.training.workers)]
    try:
        global_path = workers[0].submit(initialize, cfg.model_dump(), 0, str(path / "initial")).result()
        for device, worker in enumerate(workers[1:], 1):
            worker.submit(initialize, cfg.model_dump(), device, str(path / f"worker-{device}-initial")).result()
        contact_path = generate(cfg)
        geometry = read(contact_path)
        atomic_json(path / "contacts.json", geometry)
        contacts = geometry["contacts"]
        if cfg.network.policy == "connected":
            contacts = [{"client": i, "station": "connected-ground", "start": 0, "end": cfg.orbit.mission_seconds, "delay_seconds": 0} for i in range(cfg.training.clients)]
        atomic_json(path / "effective_contacts.json", contacts)
        link = LinkEngine(contacts, cfg.network)
        n = cfg.training.clients
        state = {
            "data_seed": cfg.seed, "training_seed": cfg.training_seed,
            "cursors": [0] * n, "age": [0] * n, "version": 0,
            "performed_steps": 0, "delivered_steps": 0, "accepted_steps": 0,
            "performed_examples": 0, "accepted_examples": 0,
            "accepted_updates": [0] * n, "accepted_digests": [[] for _ in range(n)],
            "cursor_rule": "Advance on performed work, including dropped updates; next round installs current global weights and resets AdamW",
        }
        atomic_json(path / "base_validation.json", workers[0].submit(assess, cfg.model_dump(), 0, global_path, "", False, True).result())
        fields = ["round", "mission_seconds", "version", "participants", "performed_steps", "accepted_steps", "wire_bytes", "validation_answer_loss", "actual_elapsed_seconds"]
        with (path / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for round_id in range(cfg.training.rounds):
                start = round_id * cfg.network.round_seconds
                deadline = min(start + cfg.network.round_seconds, cfg.orbit.mission_seconds)
                if start >= deadline:
                    break
                round_path = path / f"round-{round_id:04d}"
                round_path.mkdir()
                global_state = load_file(global_path)
                down_path = round_path / "global_B.safetensors"
                save_file({k: v for k, v in global_state.items() if "lora_B" in k}, str(down_path))
                old_ids = {r["id"] for r in link.receipts}
                link, tasks, predicted = plan_round(link, n, start, deadline, down_path.stat().st_size, cfg.training.device_step_seconds, cfg.training.local_steps, cfg.network.policy, state["age"], round_id)
                receipt_map = {r["id"]: r for r in link.receipts}
                current_receipts = [r for r in link.receipts if r["id"] not in old_ids]
                for receipt in current_receipts:
                    if receipt["direction"] == "download":
                        bind_receipt(receipt, down_path)
                atomic_json(round_path / "plan.json", {"tasks": tasks, "predicted": predicted})
                pending = []
                for device, task in enumerate(tasks):
                    if not receipt_map[task["download_id"]]["complete"]:
                        raise ValueError("Training scheduled before complete model delivery")
                    payload = {
                        "config": cfg.model_dump(), "device": device,
                        "output": str(round_path / f"client-{task['client']}"),
                        "client": task["client"], "cursor": state["cursors"][task["client"]],
                        "round": round_id, "version": state["version"],
                        "global_adapter": global_path, "local_steps": task["steps"],
                    }
                    pending.append((task, workers[device].submit(execute, payload)))
                results, delivered = [], []
                for task, future in pending:
                    result = future.result()
                    if result["steps"] != task["steps"] or result["cursor"] - state["cursors"][task["client"]] != task["steps"] * cfg.training.accumulation:
                        raise ValueError("Executed steps or example coverage differs from the mission plan")
                    up = receipt_map[task["upload_id"]]
                    bind_receipt(up, project_path(result["upload_path"]))
                    if up["sha256"] != result["upload_sha256"]:
                        raise ValueError("Uploaded payload changed")
                    result["receipt"] = up
                    append(path / "client_updates.jsonl", result)
                    results.append(result)
                    if up["complete"]:
                        delivered.append({**result, "tensors": load_file(result["upload_path"])})
                merged, membership, rejected = aggregate(
                    delivered, set(range(n)), state["version"], cfg.model.identity + "@" + cfg.model.revision,
                    cfg.network.min_participants,
                    expected_shapes={k: tuple(v.shape) for k, v in global_state.items() if "lora_B" in k},
                    expected_a_hash=fixed_a_hash(global_state), expected_contract=cfg.model.model_dump(),
                )
                if set(membership) != set(predicted["accepted_clients"]) or bool(merged is not None) != predicted["successful"]:
                    raise ValueError("Actual aggregation membership differs from predicted delivery")
                apply_progress(state, results, membership)
                if merged is not None:
                    global_state.update(merged)
                    global_path = str(round_path / "shared_adapter.safetensors")
                    save_file(global_state, global_path)
                    state["version"] += 1
                queue = EventQueue()
                queue.clock = start
                for task in tasks:
                    queue.push(task["ready"], "model_delivered", {"client": task["client"]})
                    queue.push(task["complete"], "training_complete", {"client": task["client"], "steps": task["steps"]})
                for receipt in current_receipts:
                    append(path / "transfers.jsonl", receipt)
                    for chunk in receipt["chunks"]:
                        queue.push(chunk["start"], "chunk_start", {"id": receipt["id"], **chunk})
                        queue.push(chunk["arrival"], "chunk_verified", {"id": receipt["id"], **chunk})
                queue.push(deadline, "round_closed", {"participants": membership, "rejected": rejected, "version": state["version"]}, 10)
                for event in queue.drain():
                    append(path / "events.jsonl", {"time": event.time, "kind": event.kind, **event.payload})
                evaluation = workers[0].submit(assess, cfg.model_dump(), 0, global_path, "", False).result()
                writer.writerow({"round": round_id, "mission_seconds": deadline, "version": state["version"], "participants": len(membership), "performed_steps": state["performed_steps"], "accepted_steps": state["accepted_steps"], "wire_bytes": link.wire_bytes, "validation_answer_loss": evaluation["validation"]["answer_loss"], "actual_elapsed_seconds": time.perf_counter() - started})
                stream.flush()
                state.update({"next_round": round_id + 1, "global_adapter": global_path, "global_sha256": digest(global_path), "link": link.snapshot(), "config_hash": cfg.fingerprint(), "source_sha256": source_identity()})
                atomic_json(round_path / "checkpoint.json", state)
                atomic_json(path / "checkpoint.json", state)
            final = workers[0].submit(assess, cfg.model_dump(), 0, global_path, str(path / "export"), True, True).result()
        atomic_json(path / "evaluation.json", final)
        atomic_json(path / "work_accounting.json", {k: v for k, v in state.items() if k not in {"link", "source_sha256"}})
        atomic_json(path / "health.json", {"status": "completed", "actual_seconds": time.perf_counter() - started, "global_version": state["version"], "quality": "validation only; no test or task benchmark claims"})
        return path
    except BaseException as error:
        atomic_json(path / "health.json", {"status": "failed", "reason": str(error)})
        raise
    finally:
        for worker in workers:
            worker.shutdown(wait=True, cancel_futures=True)
