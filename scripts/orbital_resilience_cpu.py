import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

from swarmshield.orbital.checkpoint import append, atomic_json, read
from swarmshield.orbital.config import ROOT, Settings, digest, load, project_path


def worker(run_id, fault_plan, resume):
    import torch
    from safetensors.torch import load_file, save_file
    import swarmshield.orbital.resilient_mission as mission
    import swarmshield.posttraining.trainer as trainer
    from swarmshield.orbital.runner import make_run
    from swarmshield.posttraining.model_loader import fixed_a_hash

    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Tiny CPU validation requires a CPU Slurm job with CUDA disabled")
    values = load("configs/orbital_sft/mission_repeats/seed42_connected.yaml").model_dump()
    values["training"].update(clients=2, workers=2, rounds=3, local_steps=2, accumulation=2, validation_examples=2)
    values["orbit"].update(planes=2, mission_seconds=3000)
    values["network"].update(round_seconds=1000, policy="connected", chunk_bytes=32)
    cfg = Settings.model_validate(values)
    run_path = project_path("results/orbital") / run_id

    def initialize(config, device, output, parent=None):
        path = Path(output)
        path.mkdir()
        state = load_file(parent) if parent else {"layer.lora_A": torch.ones(2), "layer.lora_B": torch.zeros(2)}
        save_file(state, str(path / "adapter.safetensors"))
        return str(path / "adapter.safetensors")

    def execute(task):
        path = Path(task["output"])
        path.mkdir()
        state = load_file(task["global_adapter"])
        weights = torch.nn.Parameter(state["layer.lora_B"].clone())
        optimizer = torch.optim.SGD([weights], lr=0.05)
        x = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
        target = torch.tensor([1.0, 2.0]) + task["client"] * 0.1
        for _ in range(task["local_steps"]):
            optimizer.zero_grad()
            loss = ((x @ weights - target) ** 2).mean()
            loss.backward()
            optimizer.step()
        save_file({"layer.lora_B": weights.detach()}, str(path / "upload.safetensors"))
        count = task["local_steps"] * cfg.training.accumulation
        result = {"client": task["client"], "version": task["version"], "base_identity": cfg.model.identity + "@" + cfg.model.revision,
                  "unique_examples": count, "example_digests": [f"synthetic-c{task['client']}-e{i}" for i in range(task["cursor"], task["cursor"] + count)],
                  "steps": task["local_steps"], "cursor": task["cursor"] + count,
                  "upload_path": str(path / "upload.safetensors"), "upload_sha256": digest(path / "upload.safetensors"),
                  "fixed_A_sha256": fixed_a_hash(state), "adapter_contract": cfg.model.model_dump()}
        append(run_path / "tiny_training_calls.jsonl", {"round": task["round"], "client": task["client"], "steps": task["local_steps"]})
        return result

    def assess(config, device, adapter, output, final=False, full_validation=False):
        value = load_file(str(adapter))
        result = {"validation": {"answer_loss": float((value["layer.lora_B"] ** 2).sum()),
                                  "scope": "Synthetic CPU fixture metric; not language-model answer loss"}}
        if final:
            path = Path(output)
            path.mkdir()
            save_file(value, str(path / "adapter_model.safetensors"))
            atomic_json(path / "inference_contract.json", {"fixture": "tiny CPU linear model", "evaluation": result})
        return result

    class Done:
        def __init__(self, value):
            self.value = value

        def result(self):
            return self.value

    class ImmediateWorker:
        def submit(self, function, *args):
            return Done(function(*args))

        def shutdown(self, **kwargs):
            pass

    def make_cpu_run(cfg, name, mode):
        return make_run(cfg, name, "tiny_cpu_recovery_validation")

    geometry = project_path("results/orbital") / f"tiny-cpu-contacts-{os.environ['SLURM_JOB_ID']}.json"
    atomic_json(geometry, {"contacts": [], "scope": "Connected synthetic CPU validation fixture"})
    mission.make_run = make_cpu_run
    mission.require_allocation = lambda cfg: None
    mission.compatibility = lambda cfg: None
    mission.generate = lambda cfg: geometry
    mission.pool = ImmediateWorker
    trainer.initialize, trainer.execute, trainer.assess = initialize, execute, assess
    mission.run_resilient_mission(cfg, run_id, resume, fault_plan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--fault-plan")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.run_id, args.fault_plan, args.resume)
        return
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run this validation through the CPU Slurm job")
    from safetensors.torch import load_file
    import torch

    identity = os.environ["SLURM_JOB_ID"]
    output = project_path("results/orbital") / f"resilience-cpu-{identity}"
    output.mkdir(exist_ok=False)
    faults = {"baseline": [], "chunk": [{"stage": "after_chunk", "round": 0, "client": 0, "chunk": 0, "direction": "upload"}],
              "eligibility": [{"stage": "after_eligibility", "round": 0, "client": 0, "direction": "upload"}],
              "before_commit": [{"stage": "before_round_commit", "round": 1}],
              "after_commit": [{"stage": "after_round_commit", "round": 1}],
              "two_interruptions": [{"stage": "after_chunk", "round": 0, "client": 0, "chunk": 0, "direction": "upload"}, {"stage": "after_round_commit", "round": 1}]}
    outcomes = []
    baseline = None
    for case, plan in faults.items():
        fault_file = output / f"{case}-faults.json"
        atomic_json(fault_file, plan)
        run_id = f"resilience-cpu-{identity}-{case}"
        run = project_path("results/orbital") / run_id
        codes = []
        while True:
            command = [sys.executable, __file__, "--worker", "--run-id", run_id, "--fault-plan", str(fault_file)]
            if codes:
                command.append("--resume")
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            (output / f"{case}-attempt-{len(codes) + 1}.log").write_text(result.stdout + result.stderr)
            codes.append(result.returncode)
            if result.returncode == 0:
                break
            if result.returncode != 75 or len(codes) > len(plan):
                raise RuntimeError(f"{case} failed unexpectedly; see {output}")
        if len(codes) != len(plan) + 1:
            raise AssertionError("Not all declared interruptions were injected")
        state = read(run / "work_accounting.json")
        fields = ("cursors", "age", "version", "performed_steps", "accepted_steps", "accepted_examples", "accepted_updates", "accepted_digests")
        progress = {k: state[k] for k in fields}
        weights = load_file(state["global_adapter"])
        checkpoint = read(run / "checkpoint.json")
        database = sqlite3.connect(run / "mission.sqlite")
        commits = database.execute("SELECT COUNT(*) FROM commits WHERE round_id>=0").fetchone()[0]
        database.close()
        if commits != 3:
            raise AssertionError("Round was committed twice or omitted")
        if baseline is None:
            baseline = (progress, weights, checkpoint["link"])
        else:
            assert progress == baseline[0] and checkpoint["link"] == baseline[2]
            assert set(weights) == set(baseline[1]) and all(torch.equal(weights[k], baseline[1][k]) for k in weights)
        previous_calls = len((run / "tiny_training_calls.jsonl").read_text().splitlines())
        inspect = subprocess.run([*command, "--resume"] if "--resume" not in command else command,
                                 capture_output=True, text=True, timeout=60)
        if inspect.returncode or len((run / "tiny_training_calls.jsonl").read_text().splitlines()) != previous_calls:
            raise AssertionError("Completed-run retry repeated local training")
        receipts = [json.loads(v) for v in (run / "transfers.jsonl").read_text().splitlines()]
        assert len(receipts) == len({r["id"] for r in receipts})
        outcomes.append({"case": case, "process_exit_codes": codes, "committed_rounds": commits,
                         "model_tensors_exactly_equal_to_baseline": True, "logical_progress_equal": True,
                         "link_reservations_equal": True, "completed_retry_did_not_train": True,
                         "local_training_calls_including_replays": previous_calls,
                         "durable_chunks_reused": sum(r.get("durable_chunks_reused", 0) for r in receipts),
                         "run": str(run.relative_to(ROOT))})
    report = {"status": "passed", "cases": outcomes,
              "scope": "Actual production resilient coordinator with tiny synthetic CPU SGD backend. Full Qwen training/compatibility gates are replaced only inside this explicit validation script.",
              "limits": ["Process exits, not power-loss or storage-failure emulation.", "Modeled mission time is replayed; no mission-time disruption penalty is measured.", "Logical work is committed once; uncheckpointed physical computation may repeat.", "CPU exact equality does not establish GPU bitwise determinism."]}
    atomic_json(output / "results.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
