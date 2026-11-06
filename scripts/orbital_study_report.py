import argparse
import csv
import hashlib
import io
import json
import random
import subprocess
from collections import defaultdict
from pathlib import Path

from swarmshield.orbital.checkpoint import atomic_json, read
from swarmshield.orbital.config import digest, project_path
from swarmshield.posttraining.data import rows


def communication(receipts):
    chunks = [chunk for receipt in receipts for chunk in receipt["chunks"]]
    uploads = [r for r in receipts if r["direction"] == "upload" and r["complete"]]
    return {
        "transfers": len(receipts),
        "complete_transfers": sum(r["complete"] for r in receipts),
        "wire_bytes": sum(c["wire_bytes"] for c in chunks),
        "verified_payload_bytes": sum(r["verified_bytes"] for r in receipts),
        "last_upload_arrival_seconds": max((r["arrival"] for r in uploads), default=None),
        "transfer_elapsed_seconds": sum(r["arrival"] - r["ready"] for r in receipts if r["complete"]),
        "station_service_seconds": sum(c["finish"] - c["start"] for c in chunks),
        "wait_before_first_chunk_seconds": sum(r["chunks"][0]["start"] - r["ready"] for r in receipts if r["chunks"]),
        "elapsed_definition": "Per-transfer elapsed includes contention and visibility waits; sum is not mission wall time",
    }


def tensor_hash(path):
    from safetensors import safe_open

    h = hashlib.sha256()
    with safe_open(str(path), framework="np") as stream:
        for name in sorted(stream.keys()):
            tensor = stream.get_tensor(name)
            h.update(name.encode())
            h.update(str((tensor.shape, tensor.dtype)).encode())
            h.update(tensor.tobytes())
    return h.hexdigest()


def accounting(job_ids):
    command = ["sacct", "-j", ",".join(map(str, job_ids)), "--parsable2", "--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocTRES,TotalCPU,MaxRSS"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    parsed = []
    for row in csv.DictReader(io.StringIO(result.stdout), delimiter="|"):
        if row["JobIDRaw"] not in set(map(str, job_ids)):
            continue
        tres = dict(part.split("=", 1) for part in row["AllocTRES"].split(",") if "=" in part)
        hours = int(row["ElapsedRaw"] or 0) / 3600
        row["allocated_gpu_hours"] = hours * int(tres.get("gres/gpu", 0))
        row["slurm_billing_weight_hours"] = hours * float(tres.get("billing", 0))
        parsed.append(row)
    return parsed, result.stdout


def matched_pair(orbital, control):
    left, right = read(orbital / "manifest.json")["config"], read(control / "manifest.json")["config"]
    defaults = {"epoch_number": 1, "full_shard_pass": False}
    keys = ("clients", "workers", "rounds", "local_steps", "accumulation", "learning_rate", "weight_decay", "grad_clip", "optimizer_reset", "schedule", "full_shard_pass", "epoch_number", "device_step_seconds")
    differences = []
    for name in ("model", "data", "seed", "orbit"):
        if left[name] != right[name]:
            differences.append(name)
    for name in keys:
        if left["training"].get(name, defaults.get(name)) != right["training"].get(name, defaults.get(name)):
            differences.append("training." + name)
    for name in set(left["network"]) | set(right["network"]):
        if name != "policy" and left["network"].get(name) != right["network"].get(name):
            differences.append("network." + name)
    initial_equal = tensor_hash(orbital / "initial/adapter.safetensors") == tensor_hash(control / "initial/adapter.safetensors")
    coverage_equal = read(orbital / "coverage.json")["processed_examples"] == read(control / "coverage.json")["processed_examples"]
    first_epoch = left["training"].get("epoch_number", 1) == 1
    return {"configuration_differences_excluding_policy_and_evaluation": differences, "identical_initial_adapter_tensors": initial_equal, "identical_initial_tensors_required": first_epoch, "equal_example_coverage": coverage_equal, "matched": not differences and (initial_equal or not first_epoch) and coverage_equal, "continuation_rule": "Epoch two follows its own policy's completed epoch one; downstream initial differences can be experimental outcomes"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plan = read(project_path(args.plan))
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    evidence, missing = {}, []
    hashes = {}
    for label, name in plan["runs"].items():
        path = project_path(name)
        if not (path / "health.json").exists():
            missing.append(label)
            continue
        item = {"path": name, "health": read(path / "health.json")}
        if item["health"]["status"] != "completed":
            missing.append(label)
        for filename in ("coverage.json", "evaluation.json", "results.json", "comparison.json", "lineage.json"):
            if (path / filename).exists():
                item[filename] = read(path / filename)
        if (path / "transfers.jsonl").exists():
            item["communication"] = communication(rows(path / "transfers.jsonl"))
        if (path / "client_updates.jsonl").exists():
            updates = rows(path / "client_updates.jsonl")
            item["training"] = {
                "supervised_tokens": sum(u["supervised_tokens"] for u in updates),
                "local_steps": sum(u["steps"] for u in updates),
                "client_peak_memory_bytes": {str(u["client"]): u["peak_memory_bytes"] for u in updates},
                "device_names": sorted({u["device_name"] for u in updates}),
                "local_training_seconds_sum": sum(u["seconds"] for u in updates),
            }
        evidence[label] = item
        for filename in ("manifest.json", "health.json", "coverage.json", "evaluation.json", "results.json", "comparison.json", "transfers.jsonl", "client_updates.jsonl", "checkpoint.json"):
            f = path / filename
            if f.exists():
                hashes[str(f.relative_to(project_path(".")))] = digest(f)
    matching = {}
    for epoch in (1, 2):
        a, b = f"orbital_epoch{epoch}", f"connected_epoch{epoch}"
        if a in evidence and b in evidence and a not in missing and b not in missing:
            matching[str(epoch)] = matched_pair(project_path(plan["runs"][a]), project_path(plan["runs"][b]))
    costs, raw = accounting(plan["jobs"])
    (output / "slurm-accounting.psv").write_text(raw)
    warnings = [
        "Single training seed and one connected/orbital pair; no repeatability distribution.",
        "Real GPU updates ran on Earth. Orbit geometry, link rate and onboard compute time are modeled assumptions.",
        "All five clients contributed under the current generous deadlines; identical quality and bytes across controls are valid outcomes.",
        "Connected control uses one idealized serial ground link; orbital uses two hypothetical ground sites. This is an explicit topology difference.",
        "Training source evolved after historical orbital runs; matching validates declared budgets and initial tensors, not identical source revisions.",
        "Adapter trains B factors with fixed A in selected language attention projections; this is a restricted adapter experiment.",
        "IFEval is a fixed 100/541 sample with a 2048-token output cap; sample scores are not full-benchmark leaderboard scores.",
        "Benchmark/pretraining contamination is unknown. Epoch-one test loss was previously exposed; final selection used validation only.",
        "GPU hours measure allocated wall time, including loading/evaluation/idle time; they are not utilization measurements or dollar charges.",
        "Slurm billing weight-hours are reported separately; no service-unit or monetary conversion is assumed.",
        "This report covers the enumerated study jobs, not unrelated account use. Scheduler records may need refreshing after settlement.",
    ]
    totals = defaultdict(float)
    for row in costs:
        totals[plan.get("job_roles", {}).get(row["JobIDRaw"], "other")] += row["allocated_gpu_hours"]
    paired = {}
    final_path = project_path(plan["runs"]["final_evaluation"])
    base_details = final_path / "ifeval-0.strict.jsonl"
    if base_details.exists():
        base_flags = [int(r["follow_all_instructions"]) for r in rows(base_details)]
        for index, name in ((1, "selected_orbital"), (2, "connected")):
            f = final_path / f"ifeval-{index}.strict.jsonl"
            if not f.exists():
                continue
            flags = [int(r["follow_all_instructions"]) for r in rows(f)]
            if len(flags) != len(base_flags):
                raise ValueError("Paired IFEval response counts differ")
            differences = [a - b for a, b in zip(flags, base_flags)]
            rng = random.Random(42)
            samples = sorted(sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(2000))
            paired[name] = {"strict_prompt_accuracy_delta": sum(differences) / len(differences), "paired_bootstrap_95": [samples[49], samples[1949]], "resamples": 2000, "seed": 42}
    payload = {"plan": plan, "incomplete_runs": missing, "matched_controls": matching, "evidence": evidence, "accounting": costs, "allocated_gpu_hours_total": sum(r["allocated_gpu_hours"] for r in costs), "allocated_gpu_hours_by_role": dict(totals), "ifeval_paired_deltas_vs_base": paired, "limitations": warnings}
    atomic_json(output / "study.json", payload)
    atomic_json(output / "evidence-sha256.json", {"plan_sha256": digest(project_path(args.plan)), "report_script_sha256": digest(project_path("scripts/orbital_study_report.py")), "evidence": hashes})
    lines = ["# Orbital federated SFT: measured study", "", f"Study completeness: {'incomplete: ' + ', '.join(missing) if missing else 'all planned runs completed'}.", "", "Five synthetic LEO clients train distinct shards on five allocated A100s. A frozen 27B base receives fixed-A/trainable-B NF4/BF16 adapters. Two complete passes cover 6,216 training conversations per epoch; 344 validation conversations select epoch two before the final 365-conversation test comparison.", "", "## Final quality", "", "| Model | Validation answer loss | Test answer loss | IFEval strict prompts | IFEval loose prompts | Capped outputs |", "|---|---:|---:|---:|---:|---:|"]
    final = evidence.get("final_evaluation", {}).get("results.json", {}).get("results", {})
    for name, item in final.items():
        score = item.get("ifeval", {})
        strict, loose = score.get("strict", {}).get("prompt_accuracy"), score.get("loose", {}).get("prompt_accuracy")
        lines.append(f"| {name} | {item['validation']['answer_loss']:.6f} | {item['test']['answer_loss']:.6f} | {strict} | {loose} | {score.get('at_token_limit')} |")
    if not final:
        lines.extend(["", "Final quality results are pending or incomplete; no outcomes are inferred."])
    if paired:
        lines.extend(["", "Paired strict-prompt accuracy deltas versus base, with seed-42 bootstrap intervals (100 prompt pairs; 2,000 resamples):", "", "```json", json.dumps(paired, indent=2), "```"])
    lines.extend(["", "## Existing development benchmarks", "", "| Model | ARC correct / 299 | GSM8K correct / 100 |", "|---|---:|---:|"])
    for name, item in evidence.get("development_evaluation", {}).get("comparison.json", {}).get("results", {}).items():
        lines.append(f"| {name} | {item['arc']['correct']} | {item['gsm8k']['correct']} |")
    lines.extend(["", "Development ARC/GSM8K scores use custom prompts/scoring with a 1,024-token GSM8K cap. They do not establish broad capability improvement.", "", "## Simulated communication", "", "| Run | Complete transfers | Wire MiB | Last upload arrival (mission seconds) | Sum of initial waits (seconds) |", "|---|---:|---:|---:|---:|"])
    for label, item in evidence.items():
        comm = item.get("communication")
        if comm:
            lines.append(f"| {label} | {comm['complete_transfers']}/{comm['transfers']} | {comm['wire_bytes'] / 2**20:.3f} | {comm['last_upload_arrival_seconds']} | {comm['wait_before_first_chunk_seconds']:.3f} |")
    lines.extend(["", "Serialized B-state sizes, chunk overhead, checksums, contact windows and receipt timestamps come from saved transfer evidence. Last upload arrival measures when all delivered updates could be available; the configured round still closes at its fixed deadline.", "", "## Control checks", "", "```json", json.dumps(matching, indent=2), "```", "", "## Allocated GPU cost", "", "| Job | State | Elapsed seconds | Allocated GPU hours |", "|---|---|---:|---:|"])
    for item in costs:
        lines.append(f"| {item['JobIDRaw']} | {item['State']} | {item['ElapsedRaw']} | {item['allocated_gpu_hours']:.4f} |")
    lines.extend(["", "Allocated GPU hours by job role:", "", "```json", json.dumps(dict(totals), indent=2), "```"])
    lines.extend(["", f"Total enumerated allocated GPU hours: **{payload['allocated_gpu_hours_total']:.4f}**. Exact scheduler records and billing weight-hours are in `slurm-accounting.psv` and `study.json`.", "", "## Limits and interpretation", ""] + [f"- {warning}" for warning in warnings])
    lines.extend(["", "## Reproduce", "", "The immutable run paths, job IDs and Slurm dependency chain are in the study plan. Model/data revisions, environments, source hashes and resource allocations are in each run manifest and its `results/orbital-accounting/JOBID` directory. IFEval questions, scorer commit, Apache license, dependency freeze and resource hashes are staged under the project. Raw model outputs and strict/loose checks remain with the final evaluation. Run `scripts/orbital_study_report.py --plan configs/orbital_sft/final_study.json --output results/orbital/final-study-v1` inside the reviewed environment to refresh this report. No training or GPU use is performed by report generation.", ""])
    (output / "report.md").write_text("\n".join(lines))
    print(output / "report.md")


if __name__ == "__main__":
    main()
