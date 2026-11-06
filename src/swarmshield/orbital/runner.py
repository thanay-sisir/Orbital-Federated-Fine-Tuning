import csv
import json
import multiprocessing
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .aggregation import aggregate
from .checkpoint import append, atomic_json, read
from .config import ROOT, digest, project_path
from .contacts import generate
from .events import EventQueue
from .network import LinkEngine


def require_allocation(cfg):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("ORBITAL_ALLOW_TRAINING") != "1":
        raise RuntimeError("Training only runs from an explicitly enabled Slurm compute allocation")
    import torch

    if torch.cuda.device_count() < cfg.training.workers:
        raise ValueError("Insufficient GPUs for independent workers")
    cfg.ready(training=True)
    from swarmshield.posttraining.data import verify_prepared

    return verify_prepared(cfg)


def environment_identity():
    import hashlib
    from importlib.metadata import distributions

    packages = sorted((d.metadata["Name"].lower(), d.version) for d in distributions())
    payload = {
        "packages": packages,
        "requirements": digest(project_path("infra/rivanna/environment/requirements-gpu.txt")),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def source_identity():
    return {
        str(p.relative_to(ROOT)): digest(p)
        for base in ("src/swarmshield/orbital", "src/swarmshield/posttraining", "infra/rivanna")
        for p in sorted((ROOT / base).rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def make_run(cfg, run_id, mode):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_id):
        raise ValueError("Invalid run identity")
    path = project_path("results/orbital") / run_id
    path.mkdir(parents=True, exist_ok=False)
    source = source_identity()
    atomic_json(
        path / "manifest.json",
        {
            "mode": mode,
            "config": cfg.model_dump(),
            "config_hash": cfg.fingerprint(),
            "source_sha256": source,
            "environment_identity": environment_identity(),
            "clocks": {
                "actual": "perf_counter seconds",
                "mission": "synthetic device profile seconds",
            },
            "physical_gpus": cfg.training.workers,
            "virtual_clients": cfg.training.clients,
        },
    )
    atomic_json(
        path / "resources.json",
        {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_ACCOUNT",
                "SLURM_JOB_PARTITION",
                "SLURM_JOB_NODELIST",
                "SLURM_CPUS_PER_TASK",
                "SLURM_MEM_PER_NODE",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    )
    for name in ("events.jsonl", "transfers.jsonl", "client_updates.jsonl"):
        (path / name).touch()
    atomic_json(path / "health.json", {"status": "running", "quality": "unmeasured"})
    return path


def sanitize_worker():
    allowed = {
        key: os.environ[key]
        for key in (
            "PATH",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
            "PYTHONNOUSERSITE",
            "PYTHONDONTWRITEBYTECODE",
            "CUDA_VISIBLE_DEVICES",
            "TMPDIR",
            "OMP_NUM_THREADS",
            "TOKENIZERS_PARALLELISM",
            "SLURM_JOB_ID",
        )
        if key in os.environ
    }
    os.environ.clear()
    os.environ.update(allowed)
    home = project_path(".cache/worker-home")
    home.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "HOME": str(home),
            "HF_HOME": str(project_path(".cache/huggingface-offline")),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        }
    )


def pool():
    return ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn"), initializer=sanitize_worker
    )


def pilot(cfg, run_id):
    from swarmshield.posttraining.trainer import execute

    require_allocation(cfg)
    path = make_run(cfg, run_id, "compatibility")
    started = time.perf_counter()
    try:
        with pool() as worker:
            result = worker.submit(
                execute,
                {
                    "config": cfg.model_dump(),
                    "device": 0,
                    "output": str(path / "pilot"),
                    "initial": True,
                    "pilot": True,
                    "client": 0,
                    "cursor": 0,
                    "round": 0,
                    "version": 0,
                },
            ).result()
        atomic_json(
            path / "compatibility.json",
            {
                "passed": True,
                "model": cfg.model.model_dump(),
                "max_length": cfg.data.max_length,
                "result": result,
                "environment_identity": environment_identity(),
                "source_sha256": source_identity(),
                "environment_sha256": digest(
                    project_path("infra/rivanna/environment/requirements-gpu.txt")
                ),
            },
        )
        atomic_json(
            path / "health.json",
            {
                "status": "completed",
                "protocol": "pilot_only",
                "quality": "answer_loss_only",
                "actual_seconds": time.perf_counter() - started,
            },
        )
    except BaseException as error:
        atomic_json(path / "health.json", {"status": "failed", "reason": str(error)})
        raise
    return path


def run(cfg, run_id, resume=None):
    if cfg.network.policy in {"feasible_fair", "retry_backoff"}:
        raise ValueError("Use mission-run for feasibility/fairness scheduling")
    from safetensors.torch import load_file, save_file

    from swarmshield.posttraining.model_loader import fixed_a_hash
    from swarmshield.posttraining.trainer import assess, execute, initialize

    require_allocation(cfg)
    gate = read(project_path(cfg.training.compatibility_run) / "compatibility.json")
    if not gate["passed"] or gate["model"] != cfg.model.model_dump():
        raise ValueError("A matching compatibility pilot must pass before federated execution")
    if gate["max_length"] < cfg.data.max_length:
        raise ValueError("Pilot context shorter than requested training context")
    if gate["environment_sha256"] != digest(
        project_path("infra/rivanna/environment/requirements-gpu.txt")
    ):
        raise ValueError("Training environment changed after pilot")
    if gate["environment_identity"] != environment_identity():
        raise ValueError("Installed package identity changed after compatibility pilot")
    if gate["source_sha256"] != source_identity():
        raise ValueError("Reviewed training source changed after compatibility pilot")
    parent_adapter, lineage = None, None
    if cfg.training.epoch_number > 1:
        from .continuation import completed_adapter

        if resume or not cfg.training.parent_run:
            raise ValueError("Epoch continuation needs a parent run and cannot also resume")
        parent_adapter, lineage = completed_adapter(
            cfg.training.parent_run, cfg, cfg.training.epoch_number - 1
        )
    elif cfg.training.parent_run:
        raise ValueError("Epoch one must start from the original base")
    path = make_run(cfg, run_id, "federated")
    if lineage:
        atomic_json(path / "lineage.json", lineage)
    workers = [pool() for _ in range(cfg.training.workers)]
    started = time.perf_counter()
    try:
        global_path = (
            workers[0].submit(initialize, cfg.model_dump(), 0, str(path / "initial"), parent_adapter).result()
        )
        for device, worker in enumerate(workers[1:], 1):
            worker.submit(
                initialize, cfg.model_dump(), device, str(path / f"worker-{device}-initial"), parent_adapter
            ).result()
        contacts_path = generate(cfg)
        shutil.copyfile(contacts_path, path / "contacts.json")
        contacts = read(contacts_path)["contacts"]
        if cfg.network.policy == "connected":
            contacts = [
                {
                    "client": i,
                    "station": "connected-ground",
                    "start": 0,
                    "end": cfg.orbit.mission_seconds,
                    "delay_seconds": 0,
                }
                for i in range(cfg.training.clients)
            ]
        link = LinkEngine(contacts, cfg.network)
        state = {
            "next_round": 0,
            "version": 0,
            "cursors": [0] * cfg.training.clients,
            "global_adapter": global_path,
            "mission_clock": 0.0,
        }
        if resume:
            checkpoint = project_path(resume)
            state = read(checkpoint)
            if state["config_hash"] != cfg.fingerprint():
                raise ValueError("Resume configuration mismatch")
            if digest(project_path(state["global_adapter"])) != state["global_sha256"]:
                raise ValueError("Resume adapter checksum mismatch")
            current_source = read(path / "manifest.json")["source_sha256"]
            if state["source_sha256"] != current_source:
                raise ValueError("Resume source mismatch")
            link.restore(state["link"])
            global_path = str(project_path(state["global_adapter"]))
        with (path / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "round",
                    "version",
                    "mission_seconds",
                    "participants",
                    "wire_bytes",
                    "supervised_tokens",
                    "validation_answer_loss",
                ],
            )
            writer.writeheader()
            for round_id in range(state["next_round"], cfg.training.rounds):
                start = round_id * cfg.network.round_seconds
                deadline = min(start + cfg.network.round_seconds, cfg.orbit.mission_seconds)
                if start >= deadline:
                    break
                round_path = path / f"round-{round_id:04d}"
                round_path.mkdir()
                global_state = load_file(global_path)
                download_path = round_path / "global_B.safetensors"
                save_file(
                    {k: v for k, v in global_state.items() if "lora_B" in k}, str(download_path)
                )
                queue = EventQueue()
                queue.clock = start
                queue.push(start, "round_open", {"round": round_id, "version": state["version"]}, 0)
                for contact in contacts:
                    if start <= contact["start"] < deadline:
                        queue.push(contact["start"], "contact_start", contact, 0)
                    if start <= contact["end"] <= deadline:
                        queue.push(contact["end"], "contact_end", contact, 0)
                order = list(range(cfg.training.clients))
                if cfg.network.policy == "contact_aware":
                    order.sort(
                        key=lambda i: (
                            min(
                                (
                                    c["end"]
                                    for c in contacts
                                    if c["client"] == i and c["end"] > start
                                ),
                                default=float("inf"),
                            ),
                            i,
                        )
                    )
                tasks, updates, total_tokens = [], [], 0
                for client in order:
                    receipt = link.transfer(
                        client,
                        download_path.stat().st_size,
                        start,
                        deadline,
                        "download",
                        f"r{round_id}-c{client}-global-v{state['version']}",
                        download_path,
                    )
                    receipt["sha256"] = digest(download_path)
                    append(path / "transfers.jsonl", receipt)
                    if not receipt["complete"]:
                        continue
                    ready = receipt["arrival"]
                    queue.push(
                        ready, "download_delivered", {"client": client, "version": state["version"]}
                    )
                    complete = ready + cfg.training.local_steps * cfg.training.device_step_seconds
                    if complete > deadline:
                        queue.push(deadline, "training_not_started_deadline", {"client": client})
                        continue
                    device = len(tasks) % len(workers)
                    task = {
                        "config": cfg.model_dump(),
                        "device": device,
                        "output": str(round_path / f"client-{client}"),
                        "client": client,
                        "cursor": state["cursors"][client],
                        "round": round_id,
                        "version": state["version"],
                        "global_adapter": global_path,
                    }
                    future = workers[device].submit(execute, task)
                    tasks.append((complete, client, future))
                for complete, client, future in sorted(tasks, key=lambda t: (t[0], t[1])):
                    result = future.result()
                    total_tokens += result["supervised_tokens"]
                    state["cursors"][client] = result["cursor"]
                    queue.push(
                        complete, "training_complete", {"client": client, "steps": result["steps"]}
                    )
                    upload = Path(result["upload_path"])
                    receipt = link.transfer(
                        client,
                        upload.stat().st_size,
                        complete,
                        deadline,
                        "upload",
                        f"r{round_id}-c{client}-v{state['version']}",
                        upload,
                    )
                    receipt["sha256"] = result["upload_sha256"]
                    append(path / "transfers.jsonl", receipt)
                    result["receipt"] = receipt
                    append(path / "client_updates.jsonl", result)
                    if receipt["complete"]:
                        if digest(upload) != result["upload_sha256"]:
                            raise ValueError("Delivered update is corrupt")
                        result["tensors"] = load_file(str(upload))
                        updates.append(result)
                        queue.push(
                            receipt["arrival"],
                            "update_delivered",
                            {"client": client, "version": state["version"]},
                        )
                merged, membership, rejected = aggregate(
                    updates,
                    set(order),
                    state["version"],
                    cfg.model.identity + "@" + cfg.model.revision,
                    cfg.network.min_participants,
                    expected_shapes={
                        k: tuple(v.shape) for k, v in global_state.items() if "lora_B" in k
                    },
                    expected_a_hash=fixed_a_hash(global_state),
                    expected_contract=cfg.model.model_dump(),
                )
                if merged is not None:
                    global_state.update(merged)
                    global_path = str(round_path / "shared_adapter.safetensors")
                    save_file(global_state, global_path)
                    state["version"] += 1
                queue.push(
                    deadline,
                    "round_closed",
                    {
                        "participants": membership,
                        "rejected": rejected,
                        "version": state["version"],
                        "failed": merged is None,
                    },
                    10,
                )
                for receipt in link.receipts:
                    for chunk in receipt["chunks"]:
                        if start <= chunk["start"] < deadline:
                            queue.push(
                                chunk["start"], "chunk_start", {"id": receipt["id"], **chunk}
                            )
                            queue.push(
                                chunk["arrival"], "chunk_verified", {"id": receipt["id"], **chunk}
                            )
                for event in queue.drain():
                    append(
                        path / "events.jsonl",
                        {"time": event.time, "kind": event.kind, **event.payload},
                    )
                assessment = (
                    workers[0].submit(assess, cfg.model_dump(), 0, global_path, "", False).result()
                )
                writer.writerow(
                    {
                        "round": round_id,
                        "version": state["version"],
                        "mission_seconds": deadline,
                        "participants": len(membership),
                        "wire_bytes": link.wire_bytes,
                        "supervised_tokens": total_tokens,
                        "validation_answer_loss": assessment["validation"]["answer_loss"],
                    }
                )
                stream.flush()
                state.update(
                    {
                        "next_round": round_id + 1,
                        "mission_clock": deadline,
                        "global_adapter": global_path,
                        "global_sha256": digest(global_path),
                        "config_hash": cfg.fingerprint(),
                        "link": link.snapshot(),
                        "source_sha256": read(path / "manifest.json")["source_sha256"],
                        "checkpoint_scope": "closed round; no active computation or transfer",
                        "client_optimizer_rule": "reset each round",
                        "rng_rule": "seed + client*100003 + round",
                        "event_queue": [],
                    }
                )
                atomic_json(round_path / "checkpoint.json", state)
                atomic_json(path / "checkpoint.json", state)
        if cfg.training.full_shard_pass:
            from swarmshield.posttraining.data import verify_prepared

            sizes = verify_prepared(cfg)["shard_examples"]
            if state["cursors"] != sizes or state["version"] != 1:
                raise ValueError("One epoch incomplete: every shard must be consumed and aggregated")
            atomic_json(path / "coverage.json", {
                "epochs": 1,
                "epoch_number": cfg.training.epoch_number,
                "shard_examples": sizes,
                "processed_examples": state["cursors"],
                "all_clients_aggregated": True,
            })
        evaluation = (
            workers[0]
            .submit(assess, cfg.model_dump(), 0, global_path, str(path / "export"), True)
            .result()
        )
        atomic_json(path / "evaluation.json", evaluation)
        atomic_json(
            path / "health.json",
            {
                "status": "completed",
                "protocol": "round_boundary_reference",
                "quality": "answer_loss_only; task capability evaluation pending",
                "actual_seconds": time.perf_counter() - started,
                "global_version": state["version"],
            },
        )
        from .reporting import report

        report(path)
        return path
    except BaseException as error:
        atomic_json(path / "health.json", {"status": "failed", "reason": str(error)})
        raise
    finally:
        for worker in workers:
            worker.shutdown(wait=True, cancel_futures=True)
