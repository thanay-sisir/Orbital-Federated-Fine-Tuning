import argparse
import csv
import json

from swarmshield.orbital.checkpoint import atomic_json, read
from swarmshield.orbital.config import digest, project_path
from swarmshield.orbital.feasibility import POLICIES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--compression", required=True)
    args = parser.parse_args()
    root = project_path(args.sweep)
    summaries = {policy: read(root / policy / "summary.json") for policy in POLICIES}
    merged = []
    for case in range(96):
        reference = summaries["fixed"][case]
        result = {key: reference[key] for key in ("case", "rate_bps", "step_seconds", "round_seconds", "stations")}
        for policy in POLICIES:
            item = summaries[policy][case]
            if item["case"] != case:
                raise ValueError("Sweep case alignment mismatch")
            for field in ("accepted_updates", "accepted_steps", "scheduled_steps", "wire_bytes", "successful_rounds", "jain_participation_fairness"):
                result[f"{policy}.{field}"] = item[field]
        merged.append(result)
    candidates = []
    for item in merged:
        control, fixed = item["connected.accepted_updates"], item["fixed.accepted_updates"]
        if control and 0.1 <= fixed / control <= 0.8 and item["fixed.successful_rounds"]:
            candidates.append((abs(fixed / control - 0.5), item["case"], item))
    shortlist, selected_inputs = [], set()
    for _, _, item in sorted(candidates, key=lambda value: value[:2]):
        key = (item["rate_bps"], item["round_seconds"], item["stations"])
        if key not in selected_inputs:
            selected_inputs.add(key)
            shortlist.append(item)
        if len(shortlist) == 3:
            break
    codec = read(project_path(args.compression) / "result.json")
    atomic_json(root / "gpu-scenario-candidates.json", {
        "status": "CPU candidates only; policies not integrated or validated in GPU training",
        "selection_rule": "Choose baseline participation ratios nearest 0.5 within [0.1,0.8], using fixed versus connected only; proposed-policy outcomes do not select cases",
        "scenarios": shortlist,
        "next": "Validate planner/trainer timing parity, partial participation state, sample accounting, and fairness before freezing GPU experiments",
    })
    with (root / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(merged[0]))
        writer.writeheader()
        writer.writerows(merged)
    lines = ["# Mission-budget feasibility and codec probe", "", "CPU-only planning on real archived orbital contacts and serialized B payload sizes. No model training, loss curves or quality gains are inferred.", "", "Four scheduling policies ran over 96 identical parameter combinations each: four link rates, three assumed compute speeds, four aggregation intervals and one/two orbital ground sites. Connected control always uses one idealized serial ground link. Minimum aggregation membership is two clients; each client has a maximum of four local steps per round. Feasible/fair admission may reduce this to two or one, and therefore planned work differs across policies.", "", "## Candidate GPU scenarios", "", "Selected using baseline pressure alone, not whether the proposed policy wins.", "", "| Case | Link bit/s | Assumed seconds/step | Round seconds | Orbital sites | Fixed accepted updates | Feasible/fair accepted updates |", "|---|---:|---:|---:|---:|---:|---:|"]
    for item in shortlist:
        lines.append(f"| {item['case']} | {item['rate_bps']} | {item['step_seconds']} | {item['round_seconds']} | {item['stations']} | {item['fixed.accepted_updates']} | {item['feasible_fair.accepted_updates']} |")
    if not shortlist:
        lines.extend(["", "No cases met the prespecified baseline-pressure rule. Revise the sweep transparently; no winning scenario is assumed."])
    lines.extend(["", "## Codec probe", "", "Real int8 B-delta payloads include shapes, scales and checksums. Error feedback remains local. These historical deltas were not trained through quantized aggregation; numerical reconstruction is not a model-quality result.", "", "```json", json.dumps(codec, indent=2), "```", "", "## Limits and next gate", "", "Compute profiles and radio rates are sensitivity assumptions, not validated flight hardware. The planner uses predictable contacts and atomic admission of feasible transfer pairs. Baselines retain the current reference's download-first ordering. Transfers can resume within a round; pending chunks are not carried into a new round. Training runtime, asynchronous stale updates, power/thermal limits, packet errors and onboard hardware are not modeled here. Multi-round planning assumes fixed tensor shapes and sizes; it does not reuse historical gradients as a learning curve.", "", "Before GPU runs: couple the scheduling policy to actual executed local steps and receipt gates, verify failed-round and client-cursor handling, freeze candidate cases and evaluation settings, then run a fresh compatibility pilot. Compare quality at fixed mission budgets, report actual accepted work, and repeat selected conditions across seeds. Full expensive benchmarks run only on the final selected conditions.", ""])
    (root / "report.md").write_text("\n".join(lines))
    atomic_json(root / "evidence-sha256.json", {"summaries": {policy: digest(root / policy / "summary.json") for policy in POLICIES}, "codec_result_sha256": digest(project_path(args.compression) / "result.json")})
    print(root / "report.md")


if __name__ == "__main__":
    main()
