import argparse
from collections import defaultdict
import itertools
import json
import os
import statistics

from swarmshield.orbital.checkpoint import atomic_json, read
from swarmshield.orbital.config import digest, load, project_path
from swarmshield.orbital.feasibility import POLICIES, simulate
from swarmshield.posttraining.data import rows


def compressed():
    policy = POLICIES[int(os.environ["SLURM_ARRAY_TASK_ID"])]
    output = project_path("results/orbital/compressed-feasibility-v1") / policy
    output.mkdir(parents=True, exist_ok=False)
    codec_file = project_path("results/orbital/codec-clients-22453177/results.json")
    codec = read(codec_file)
    size = max(item["int8_serialized_bytes"] for item in codec["results"])
    cfg = load("configs/orbital_sft/five_gpu_one_epoch.yaml")
    contacts_file = project_path("results/orbital/reference-22367142/contacts.json")
    contacts = read(contacts_file)["contacts"]
    atomic_json(output / "manifest.json", {
        "policy": policy, "representative_payload_bytes": size,
        "codec_results_sha256": digest(codec_file), "contacts_sha256": digest(contacts_file),
        "script_sha256": digest(project_path("scripts/orbital_parallel_analysis.py")),
        "limits": "Largest observed int8 payload among ten archived updates, not a guaranteed bound on future sizes; compression assumed on both directions for sensitivity only; no compressed GPU training or quality estimate",
    })
    values = []
    for index, (rate, step, interval, stations) in enumerate(itertools.product((64000, 256000, 1000000, 10000000), (15, 60, 180), (1800, 3600, 7200, 14400), (1, 2))):
        names = {site.name for site in cfg.orbit.stations[:stations]}
        net = cfg.network.model_copy(update={"rate_bps": rate, "min_participants": 2})
        result = simulate([c for c in contacts if c["station"] in names], net, 5, 86400, interval, size, step, policy=policy)
        baseline = read(project_path(f"results/orbital/mission-feasibility-v1/{policy}/case-{index:03d}.json"))
        values.append({"case": index, "policy": policy, "rate_bps": rate, "step_seconds": step, "round_seconds": interval, "orbital_stations": stations, "raw_accepted_updates": baseline["accepted_updates"], "int8_planned_accepted_updates": result["accepted_updates"], "raw_accepted_steps": baseline["accepted_steps"], "int8_planned_accepted_steps": result["accepted_steps"], "raw_wire_bytes": baseline["wire_bytes"], "int8_planned_wire_bytes": result["wire_bytes"]})
    atomic_json(output / "comparison.json", values)


def compressed_report():
    output = project_path("results/orbital/compressed-feasibility-v1")
    comparisons = {policy: read(output / policy / "comparison.json") for policy in POLICIES}
    atomic_json(output / "comparison.json", comparisons)
    lines = ["# Compressed-payload planning sensitivity", "", "CPU planning only. Uses a representative serialized payload size from the archived codec probe in both directions. Actual future codec sizes, error-feedback state handling and compressed learning quality still need runtime validation.", "", "## Frozen case 62", "", "| Policy | Raw accepted updates | Int8 planned updates | Raw accepted steps | Int8 planned steps | Raw wire MiB | Int8 planned wire MiB |", "|---|---:|---:|---:|---:|---:|---:|"]
    for policy in POLICIES:
        item = comparisons[policy][62]
        lines.append(f"| {policy} | {item['raw_accepted_updates']} | {item['int8_planned_accepted_updates']} | {item['raw_accepted_steps']} | {item['int8_planned_accepted_steps']} | {item['raw_wire_bytes'] / 2**20:.3f} | {item['int8_planned_wire_bytes'] / 2**20:.3f} |")
    lines.extend(["", "All 384 matched parameter comparisons are preserved in comparison.json. Increased update counts do not prove better model quality. Current uncompressed repeated-seed GPU jobs and their frozen inputs are unchanged.", ""])
    (output / "report.md").write_text("\n".join(lines))


def instruction_audit():
    source = project_path("results/orbital/final-evaluation-22428040")
    output = project_path("results/orbital/instruction-audit-v1")
    output.mkdir(parents=True, exist_ok=False)
    base, adapter = rows(source / "ifeval-0.strict.jsonl"), rows(source / "ifeval-1.strict.jsonl")
    if len(base) != len(adapter):
        raise ValueError("Mismatched instruction evaluation sample")
    tally, regressions, improvements = defaultdict(lambda: {"instances": 0, "base_pass": 0, "adapter_pass": 0}), [], []
    for before, after in zip(base, adapter):
        if before["prompt"] != after["prompt"] or before["instruction_id_list"] != after["instruction_id_list"]:
            raise ValueError("Unpaired instruction evaluation")
        for name, old, new in zip(before["instruction_id_list"], before["follow_instruction_list"], after["follow_instruction_list"]):
            tally[name]["instances"] += 1
            tally[name]["base_pass"] += int(old)
            tally[name]["adapter_pass"] += int(new)
        evidence = {"prompt": before["prompt"], "instruction_ids": before["instruction_id_list"], "base_response": before["response"], "adapter_response": after["response"], "base_checks": before["follow_instruction_list"], "adapter_checks": after["follow_instruction_list"]}
        if before["follow_all_instructions"] and not after["follow_all_instructions"]:
            regressions.append(evidence)
        if not before["follow_all_instructions"] and after["follow_all_instructions"]:
            improvements.append(evidence)
    generated = {}
    for label, number in (("base", 0), ("orbital_adapter", 1)):
        responses = rows(source / f"responses-{number}.jsonl")
        generated[label] = {"mean_generated_tokens": statistics.mean(item["generated_tokens"] for item in responses), "capped_outputs": sum(item["at_token_limit"] for item in responses)}
    result = {"scope": "Posthoc diagnosis of the existing 100-prompt sample; not a new independent evaluation", "per_instruction": dict(tally), "lost_prompt_passes": len(regressions), "gained_prompt_passes": len(improvements), "response_lengths": generated, "regressions": regressions, "improvements": improvements, "inputs_sha256": {f"ifeval-{i}.strict.jsonl": digest(source / f"ifeval-{i}.strict.jsonl") for i in (0, 1)}}
    atomic_json(output / "audit.json", result)
    lines = ["# Instruction-following diagnosis", "", f"On the existing paired 100-prompt sample: {len(regressions)} base passes were lost and {len(improvements)} previously failed prompts became passes. This is a diagnostic sample, not a fresh test or a significance claim.", "", "| Instruction | Instances | Base passes | Adapter passes |", "|---|---:|---:|---:|"]
    for name, item in sorted(tally.items()):
        lines.append(f"| {name} | {item['instances']} | {item['base_pass']} | {item['adapter_pass']} |")
    lines.extend(["", "Raw paired examples and generation lengths are in audit.json. Do not insert evaluation prompts into training or change the frozen repeated-seed experiment based on these results. Any later recipe tuning makes this sample development evidence; use untouched evaluation prompts for a final check.", ""])
    (output / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("compressed", "compressed-report", "instruction-audit"))
    mode = parser.parse_args().mode
    {"compressed": compressed, "compressed-report": compressed_report, "instruction-audit": instruction_audit}[mode]()
