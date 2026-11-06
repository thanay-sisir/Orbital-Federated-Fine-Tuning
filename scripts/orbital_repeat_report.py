import argparse
import csv
import io
import json
import statistics
import subprocess

from swarmshield.orbital.checkpoint import atomic_json, read
from swarmshield.orbital.config import digest, project_path


def summarize(values):
    return {"n": len(values), "mean": statistics.mean(values), "sample_std": statistics.stdev(values) if len(values) > 1 else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = read(project_path(args.plan))
    output = project_path("results/orbital/mission-repeats-v1")
    output.mkdir(parents=True, exist_ok=True)
    results, missing, initial = [], [], {}
    for entry in plan["runs"]:
        path = project_path(entry["path"])
        name = f"seed{entry['seed']}/{entry['policy']}"
        if not (path / "health.json").exists() or read(path / "health.json")["status"] != "completed":
            missing.append(name)
            continue
        state = read(path / "work_accounting.json")
        final = read(path / "evaluation.json")
        manifest = read(path / "manifest.json")
        if state["data_seed"] != 42 or state["training_seed"] != entry["seed"]:
            raise ValueError("Data/training seed provenance mismatch")
        if manifest["config_hash"] != entry["config_hash"]:
            raise ValueError("Run configuration differs from frozen repeat plan")
        initial[name] = digest(path / "initial/adapter.safetensors")
        with (path / "metrics.csv").open() as stream:
            curve = list(csv.DictReader(stream))
        results.append({**entry, "validation_loss": final["validation"]["answer_loss"], "performed_steps": state["performed_steps"], "accepted_steps": state["accepted_steps"], "accepted_updates": sum(state["accepted_updates"]), "wire_bytes": int(curve[-1]["wire_bytes"]), "performed_examples": state["performed_examples"], "accepted_examples": state["accepted_examples"]})
    policies = ("connected", "fixed", "feasible_fair")
    by_policy = {}
    for policy in policies:
        values = [r for r in results if r["policy"] == policy]
        if values:
            by_policy[policy] = {field: summarize([r[field] for r in values]) for field in ("validation_loss", "performed_steps", "accepted_steps", "accepted_updates", "wire_bytes")}
    paired = []
    controls = {}
    for seed in (42, 43, 44):
        values = {r["policy"]: r for r in results if r["seed"] == seed}
        hashes = [initial[f"seed{seed}/{p}"] for p in policies if f"seed{seed}/{p}" in initial]
        controls[str(seed)] = {"all_policies_present": len(hashes) == 3, "same_initial_adapters": len(hashes) == 3 and len(set(hashes)) == 1}
        if "fixed" in values and "feasible_fair" in values:
            paired.append({"seed": seed, "validation_loss_delta_feasible_minus_fixed": values["feasible_fair"]["validation_loss"] - values["fixed"]["validation_loss"], "accepted_steps_delta": values["feasible_fair"]["accepted_steps"] - values["fixed"]["accepted_steps"]})
    seed_hashes = [initial[f"seed{s}/connected"] for s in (42, 43, 44) if f"seed{s}/connected" in initial]
    distinct_initializations = len(seed_hashes) == 3 and len(set(seed_hashes)) == 3
    raw = subprocess.run(["sacct", "-j", str(plan["array_job"]), "--parsable2", "--format=JobID,State,ExitCode,ElapsedRaw,AllocTRES,MaxRSS"], capture_output=True, text=True, check=True).stdout
    (output / "slurm-accounting.psv").write_text(raw)
    gpu_hours = 0.0
    for row in csv.DictReader(io.StringIO(raw), delimiter="|"):
        if "." not in row["JobID"]:
            tres = dict(part.split("=", 1) for part in row["AllocTRES"].split(",") if "=" in part)
            gpu_hours += int(row["ElapsedRaw"] or 0) / 3600 * int(tres.get("gres/gpu", 0))
    codec_path = project_path(plan["codec_run"]) / "results.json"
    codec = read(codec_path) if codec_path.exists() else {"status": "not available"}
    comparison = {"incomplete": missing, "runs": results, "by_policy": by_policy, "paired_differences": paired, "initialization_controls": controls, "distinct_seed_initializations": distinct_initializations, "training_array_gpu_hours": gpu_hours, "codec_probe": codec, "limits": "Three training seeds, one geometry, fixed data split/order. Means and sample standard deviations; no significance or flight-hardware claim. Positive loss delta means feasible/fair is worse."}
    atomic_json(output / "comparison.json", comparison)
    atomic_json(output / "evidence-sha256.json", {"plan_sha256": digest(project_path(args.plan)), "initial_adapters": initial, "report_script_sha256": digest(project_path("scripts/orbital_repeat_report.py"))})
    lines = ["# Repeated-seed mission-budget study", "", f"Incomplete runs: {missing if missing else 'none'}.", "", "Three fresh training seeds with immutable data seed 42. Each seed shares the same initial adapter across all policies. Different seeds must have different initial adapters. Source, geometry and maximum budgets are fixed. This study varies adapter initialization/training RNG; it does not resample data splits or orbit geometry.", "", "| Policy | Runs | Full validation loss mean | Sample std | Accepted steps mean | Performed steps mean | Wire MiB mean |", "|---|---:|---:|---:|---:|---:|---:|"]
    for policy, metrics in by_policy.items():
        loss = metrics["validation_loss"]
        lines.append(f"| {policy} | {loss['n']} | {loss['mean']:.6f} | {loss['sample_std']} | {metrics['accepted_steps']['mean']} | {metrics['performed_steps']['mean']} | {metrics['wire_bytes']['mean'] / 2**20:.3f} |")
    lines.extend(["", "## Paired comparisons", "", "Positive validation-loss differences mean the feasibility/fairness policy is worse than fixed scheduling. Do not infer a quality win from accepted-update counts alone.", "", "```json", json.dumps(paired, indent=2), "```", "", f"Training-array allocated GPU hours: {gpu_hours:.4f}; separate pilot and CPU costs are excluded.", "", f"Distinct initializations across all three seeds: {distinct_initializations}. Per-seed matching controls: `{controls}`.", "", "The all-client codec probe is preserved in comparison.json as a separate numerical experiment. It does not establish quantized training quality. Repeated runs use validation only and do not tune on the held-out test split. Three seeds provide an initial variability estimate, not a strong statistical significance claim. See the frozen plan and raw run artifacts for provenance.", ""])
    (output / "report.md").write_text("\n".join(lines))
    print(output / "report.md")


if __name__ == "__main__":
    main()
