import argparse
import csv
import io
import json
import sqlite3
import subprocess

from swarmshield.orbital.checkpoint import atomic_json, read
from swarmshield.orbital.config import digest, project_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = read(project_path(args.plan))
    output = project_path("results/orbital/resilience-study-v1")
    output.mkdir(parents=True, exist_ok=True)
    runs, missing = [], []
    for name, run_id in plan["gpu_runs"].items():
        path = project_path("results/orbital") / run_id
        health = read(path / "health.json") if (path / "health.json").exists() else {}
        if health.get("status") != "completed":
            missing.append(name)
            continue
        state = read(path / "work_accounting.json")
        database = sqlite3.connect(path / "mission.sqlite")
        commits = database.execute("SELECT COUNT(*) FROM commits WHERE round_id>=0").fetchone()[0]
        attempts = database.execute("SELECT id,status FROM attempts ORDER BY id").fetchall()
        database.close()
        receipts = [json.loads(line) for line in (path / "transfers.jsonl").read_text().splitlines()]
        metric_rows = list(csv.DictReader((path / "metrics.csv").open()))
        runs.append({"name": name, "path": str(path), "state": state, "checkpoint": read(path / "checkpoint.json"),
                     "evaluation": read(path / "evaluation.json"), "initial_adapter_sha256": digest(path / "initial_committed.safetensors"),
                     "manifest": read(path / "manifest.json"), "committed_rounds": commits, "attempts": attempts,
                     "metric_rows": len(metric_rows), "unique_transfer_ids": len({r["id"] for r in receipts}) == len(receipts),
                     "durable_chunks_reused": sum(r.get("durable_chunks_reused", 0) for r in receipts),
                     "durable_upload_chunks_reused": sum(r.get("durable_chunks_reused", 0) for r in receipts if r["direction"] == "upload"),
                     "consumed_faults": read(path / "faults_consumed.json") if (path / "faults_consumed.json").exists() else [],
                     "process_outcomes": [json.loads(v) for v in (path / "process_outcomes.jsonl").read_text().splitlines()]})
    checks = {}
    if not missing:
        import torch
        from safetensors.torch import load_file

        by_name = {r["name"]: r for r in runs}
        normal, resumed = by_name["baseline"], by_name["interrupted"]
        fields = ("cursors", "age", "version", "performed_steps", "delivered_steps", "accepted_steps", "performed_examples", "accepted_examples", "accepted_updates", "accepted_digests")
        checks["identical_logical_progress"] = all(normal["state"][k] == resumed["state"][k] for k in fields)
        checks["identical_initial_adapter"] = normal["initial_adapter_sha256"] == resumed["initial_adapter_sha256"]
        checks["same_config_source_environment"] = all(normal["manifest"][k] == resumed["manifest"][k] for k in ("config_hash", "source_sha256", "environment_identity"))
        checks["identical_link_reservations"] = normal["checkpoint"]["link"] == resumed["checkpoint"]["link"]
        checks["one_record_per_round"] = all(r["committed_rounds"] == r["metric_rows"] == 2 and r["unique_transfer_ids"] for r in runs)
        checks["two_declared_process_interruptions"] = len(resumed["consumed_faults"]) == 2 and [v["exit_code"] for v in resumed["process_outcomes"]] == [75, 75, 0]
        checks["saved_upload_chunks_reused"] = resumed["durable_upload_chunks_reused"] > 0
        left, right = (load_file(r["state"]["global_adapter"]) for r in (normal, resumed))
        checks["same_tensor_keys"] = set(left) == set(right)
        checks["fixed_A_exactly_equal"] = checks["same_tensor_keys"] and all(torch.equal(left[k], right[k]) for k in left if "lora_A" in k)
        max_error = max(float((left[k] - right[k]).abs().max()) for k in left) if checks["same_tensor_keys"] else None
        checks["adapter_equal_within_tolerance"] = checks["same_tensor_keys"] and all(torch.allclose(left[k], right[k], atol=1e-6, rtol=1e-5) for k in left)
        loss_delta = resumed["evaluation"]["validation"]["answer_loss"] - normal["evaluation"]["validation"]["answer_loss"]
        checks["validation_loss_within_tolerance"] = abs(loss_delta) <= 1e-4
    else:
        max_error = loss_delta = None
    raw = subprocess.run(["sacct", "-j", str(plan["gpu_array_job"]), "--parsable2", "--format=JobID,State,ExitCode,ElapsedRaw,AllocTRES"], text=True, capture_output=True, check=True).stdout
    (output / "slurm-accounting.psv").write_text(raw)
    gpu_hours = 0.0
    for row in csv.DictReader(io.StringIO(raw), delimiter="|"):
        if "." not in row["JobID"]:
            tres = dict(v.split("=", 1) for v in row["AllocTRES"].split(",") if "=" in v)
            gpu_hours += int(row["ElapsedRaw"] or 0) / 3600 * int(tres.get("gres/gpu", 0))
    result = {"status": "passed" if not missing and all(checks.values()) else "incomplete_or_failed", "missing": missing,
              "checks": checks, "max_adapter_absolute_difference": max_error, "validation_loss_delta": loss_delta,
              "tolerances": {"tensor_atol": 1e-6, "tensor_rtol": 1e-5, "validation_loss_atol": 1e-4},
              "gpu_array_allocated_hours_excluding_pilot": gpu_hours, "runs": runs,
              "cpu_study": read(project_path(plan["cpu_report"])),
              "scope": "Coordinator/receiver process recovery with mission-clock replay. No RF failure, power-loss, optimizer-step checkpoint or zero-overhead claim."}
    atomic_json(output / "results.json", result)
    lines = ["# Integrated mission recovery", "", f"Status: {result['status']}", "", f"Missing runs: {missing}", "",
             "| Check | Passed |", "|---|---|"]
    lines.extend(f"| {k} | {v} |" for k, v in checks.items())
    lines.extend(["", f"Maximum adapter absolute difference: {max_error}.", f"Full-validation loss delta: {loss_delta}.",
                  f"Array allocated GPU hours, excluding pilot: {gpu_hours:.4f}.", "",
                  "CPU uses a tiny synthetic SGD backend. GPU runs use the actual pinned language model, five replicas, two rounds and one local optimizer step per client per round. Faulted and uninterrupted runs share configuration, source and initial state. Modeled mission time is replayed; uncommitted physical computation can repeat and Slurm accounts for allocation overhead.", ""])
    (output / "report.md").write_text("\n".join(lines))
    print(output / "report.md")
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
