import argparse
import csv
import io
import json
import subprocess

from swarmshield.orbital.checkpoint import atomic_json, read
from swarmshield.orbital.config import digest, project_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="configs/orbital_sft/outages_study.json")
    args = parser.parse_args()
    plan = read(project_path(args.plan))
    selection = read(project_path("configs/orbital_sft/outages_selection.json"))
    output = project_path("results/orbital/outages-v1")
    results, missing, initials, sources, shared = [], [], [], [], []
    checks = {}
    for entry in selection["gpu_runs"]:
        path = project_path("results/orbital") / entry["run_id"]
        health = read(path / "health.json") if (path / "health.json").exists() else {}
        if health.get("status") != "completed":
            missing.append(entry["label"])
            continue
        state = read(path / "work_accounting.json")
        checkpoint = read(path / "checkpoint.json")
        execution = read(path / "outage_execution.json")
        manifest = read(path / "manifest.json")
        with (path / "metrics.csv").open() as stream:
            curve = list(csv.DictReader(stream))
        expected = entry["expected_cpu"]
        observed = {"performed_steps": state["performed_steps"], "accepted_steps": state["accepted_steps"],
                    "accepted_updates": sum(state["accepted_updates"]), "wire_bytes": int(curve[-1]["wire_bytes"]),
                    "failed_attempts": state["failed_transfer_attempts"]}
        per_run = {"matches_cpu_" + k: observed[k] == expected[k] for k in observed}
        per_run["matches_per_client_participation"] = state["accepted_updates"] == expected["accepted_per_client"]
        per_run["mission_deadline_preserved"] = state["mission_clock"] == checkpoint["link"]["mission_clock"] == 86400
        per_run["declared_configuration"] = manifest["config_hash"] == entry["config_hash"]
        per_run["shared_frozen_source"] = manifest["source_sha256"] == plan["source_sha256"]
        checks[entry["label"]] = per_run
        initials.append(digest(path / "initial_committed.safetensors"))
        sources.append(manifest["source_sha256"])
        config = manifest["config"]
        config["network"].pop("policy")
        shared.append(config)
        results.append({"label": entry["label"], "run_id": entry["run_id"], "policy": entry["policy"],
                        **observed, "accepted_per_client": state["accepted_updates"],
                        "validation_loss": read(path / "evaluation.json")["validation"]["answer_loss"],
                        "wasted_simulated_compute_seconds": state["wasted_simulated_compute_seconds"],
                        "observed_outage_events": len(execution["observations"]), "initial_sha256": initials[-1]})
    controls = {"same_initial_adapters": len(initials) == 4 and len(set(initials)) == 1,
                "same_source": len(sources) == 4 and all(v == sources[0] for v in sources),
                "same_config_except_policy": len(shared) == 4 and all(v == shared[0] for v in shared)}
    raw = subprocess.run(["sacct", "-j", str(plan["gpu_array_job"]), "--parsable2", "--format=JobID,State,ExitCode,ElapsedRaw,AllocTRES"], text=True, capture_output=True, check=True).stdout
    (output / "gpu-accounting.psv").write_text(raw)
    hours = 0.0
    for row in csv.DictReader(io.StringIO(raw), delimiter="|"):
        if "." not in row["JobID"]:
            tres = dict(v.split("=", 1) for v in row["AllocTRES"].split(",") if "=" in v)
            hours += int(row["ElapsedRaw"] or 0) / 3600 * int(tres.get("gres/gpu", 0))
    passed = not missing and all(controls.values()) and all(all(v.values()) for v in checks.values())
    report = {"status": "passed" if passed else "incomplete_or_failed", "missing": missing,
              "controls": controls, "conformance_checks": checks, "results": results,
              "allocated_gpu_hours_excluding_pilot": hours,
              "scope": "Four actual five-GPU trajectories, one selected synthetic outage scenario and one training seed. CPU counts must match actual work/delivery. No broad quality-win, flight-calibration or replicated quality claim."}
    atomic_json(output / "gpu_results.json", report)
    lines = ["# Training under mission-time outages", "", f"Status: {report['status']}", "", f"Missing: {missing}", "",
             "| Condition | Performed steps | Accepted steps | Updates | Failed attempts | Wire MiB | Full validation loss |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['label']} | {row['performed_steps']} | {row['accepted_steps']} | {row['accepted_updates']} | {row['failed_attempts']} | {row['wire_bytes']/2**20:.3f} | {row['validation_loss']:.6f} |")
    lines.extend(["", f"Shared controls: {controls}", "", f"Allocated GPU hours excluding pilot: {hours:.4f}.", "",
        "All conditions share geometry, maximum work, prepared data, initialization and source. The no-outage fixed control removes exogenous failures; other conditions share one realized trace. Outages advance modeled time; verified chunks survive and failed attempts consume wire bytes. Incomplete optimizer-step work is simulated overhead, not GPU-executed partial gradients. Quality is measured on all 344 validation conversations; no held-out test tuning. One training seed and selected scenario limit generalization.", ""])
    (output / "report.md").write_text("\n".join(lines))
    evidence = project_path("docs/orbital_sft/evidence")
    atomic_json(evidence / "outages_gpu_results.json", report)
    (evidence / "outages_gpu_report.md").write_text("\n".join(lines))
    plan["gpu_status"] = report["status"]
    plan["gpu_results"] = "results/orbital/outages-v1/gpu_results.json"
    plan["allocated_gpu_hours_excluding_pilot"] = hours
    atomic_json(project_path(args.plan), plan)
    document = project_path("docs/orbital_sft/OUTAGE_STUDY.md")
    text = document.read_text().split("\n## Automatic GPU result\n")[0]
    text = text.replace("GPU outcomes remain pending.", f"GPU outcome status: **{report['status']}**; see the [automatic GPU report](evidence/outages_gpu_report.md).")
    document.write_text(text + "\n## Automatic GPU result\n\n" + "\n".join(lines[2:]) + "\n")
    print(output / "report.md")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
