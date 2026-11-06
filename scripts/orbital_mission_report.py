import argparse
import csv
import io
import json
import subprocess

from swarmshield.orbital.checkpoint import atomic_json, read
from swarmshield.orbital.config import digest, project_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = read(project_path(args.plan))
    output = project_path("results/orbital/mission-case62-v1")
    output.mkdir(parents=True, exist_ok=True)
    results, missing, initial_hashes = {}, [], {}
    for policy, run_dir in plan["runs"].items():
        path = project_path(run_dir)
        if not (path / "health.json").exists():
            missing.append(policy)
            continue
        health = read(path / "health.json")
        item = {"health": health, "run": run_dir}
        initial = path / "initial/adapter.safetensors"
        if initial.exists():
            initial_hashes[policy] = digest(initial)
        if health["status"] != "completed":
            missing.append(policy)
        for name in ("evaluation.json", "work_accounting.json"):
            if (path / name).exists():
                item[name] = read(path / name)
        if (path / "metrics.csv").exists():
            item["curve"] = list(csv.DictReader((path / "metrics.csv").open()))
        results[policy] = item
    command = ["sacct", "-j", str(plan["array_job"]), "--parsable2", "--format=JobID,State,ExitCode,ElapsedRaw,AllocTRES,MaxRSS"]
    raw = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    (output / "slurm-accounting.psv").write_text(raw)
    costs = []
    for row in csv.DictReader(io.StringIO(raw), delimiter="|"):
        if "." in row["JobID"]:
            continue
        fields = dict(part.split("=", 1) for part in row["AllocTRES"].split(",") if "=" in part)
        row["allocated_gpu_hours"] = int(row["ElapsedRaw"] or 0) / 3600 * int(fields.get("gres/gpu", 0))
        costs.append(row)
    lines = ["# Coupled mission-budget GPU comparison", "", f"Incomplete policies: {missing if missing else 'none'}.", "", "One seed, one constrained case; real GPU updates with byte-bound planned transfers. This is an integration experiment, not a repeated-seed capability claim. All runs start from the original base; they are not full epochs.", "", "| Policy | Performed steps | Delivered steps | Accepted steps | Accepted updates | Full validation loss | Wire MiB |", "|---|---:|---:|---:|---:|---:|---:|"]
    for policy, result in results.items():
        state = result.get("work_accounting.json")
        final = result.get("evaluation.json")
        if not state or not final:
            continue
        wire = int(result["curve"][-1]["wire_bytes"])
        lines.append(f"| {policy} | {state['performed_steps']} | {state['delivered_steps']} | {state['accepted_steps']} | {sum(state['accepted_updates'])} | {final['validation']['answer_loss']:.6f} | {wire / 2**20:.3f} |")
    lines.extend(["", "Per-round validation uses the same 32 fixed development conversations. Final validation covers all 344. Performed work advances client cursors even if its update is dropped; accepted-example and unique-digest counts separately record incorporated work. Checkpoints are closed-round evidence; this runner does not implement automatic resume.", "", "This policy may spend more bytes and different actual compute to deliver more useful updates. Compare those costs, rather than assuming a free improvement. Radio rate and onboard compute speed remain assumptions; GPU wall time is measured separately. No compression is enabled in this uncompressed comparison.", "", "## GPU allocation", "", "```json", json.dumps(costs, indent=2), "```", "", "## Reproduce", "", "See `configs/orbital_sft/mission_gpu_study.json` for the frozen case, source/config hashes, policy order, Slurm IDs and array concurrency limit. Each run includes actual executed steps, payload/chunk hashes, events, checkpoints, examples consumed versus accepted, and evaluation. CPU planning alone is not evidence of learning. This report does not run expensive generation benchmarks or use held-out test data.", ""])
    matched_initial = len(initial_hashes) == len(plan["runs"]) and len(set(initial_hashes.values())) == 1
    total_gpu_hours = sum(item["allocated_gpu_hours"] for item in costs)
    lines.extend([f"Initial serialized adapter equality across all policies: {matched_initial}.", "", f"Training-array allocated GPU hours: {total_gpu_hours:.4f}; this excludes the separate compatibility pilot and CPU jobs.", ""])
    (output / "report.md").write_text("\n".join(lines))
    atomic_json(output / "comparison.json", {"plan": plan, "incomplete": missing, "results": results, "costs": costs, "training_array_gpu_hours": total_gpu_hours, "initial_adapter_sha256": initial_hashes, "same_initial_serialized_adapters": matched_initial})
    atomic_json(output / "evidence-sha256.json", {"plan_sha256": digest(project_path(args.plan)), "metrics": {policy: digest(project_path(item["run"]) / "metrics.csv") for policy, item in results.items() if (project_path(item["run"]) / "metrics.csv").exists()}})
    print(output / "report.md")


if __name__ == "__main__":
    main()
