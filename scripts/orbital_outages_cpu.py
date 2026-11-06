import argparse
import itertools
import json
import random
import statistics
from pathlib import Path

from swarmshield.orbital.checkpoint import atomic_json, read
from swarmshield.orbital.config import ROOT, Settings, digest, load, project_path
from swarmshield.orbital.contacts import generate
from swarmshield.orbital.outages import OutageEngine, audit, plan_outage_round
from swarmshield.orbital.runner import source_identity


POLICIES = ("fixed", "retry_backoff", "feasible_fair")
OUTPUT = project_path("results/orbital/outages-v1")


def make_trace(profile, seed, stations, clients, mission):
    rng = random.Random(seed)
    events = []
    counts = {"none": (0, 0, 0), "low": (12, 2, 4), "high": (48, 8, 16)}[profile]
    for kind, count, duration_range in (("link", counts[0], (20, 120)), ("client", counts[1], (120, 900)), ("station", counts[2], (300, 1200))):
        for target in (stations if kind == "station" else range(clients)):
            for _ in range(count):
                duration = rng.uniform(*duration_range)
                start = rng.uniform(0, mission - duration)
                events.append({"kind": kind, "station" if kind == "station" else "client": target,
                               "start": start, "end": start + duration})
    return {"schema_version": 1, "seed": seed, "profile": profile,
            "events": sorted(events, key=lambda e: (e["start"], e["kind"]))}


def simulate(cfg, contacts, trace):
    engine = OutageEngine(contacts, cfg.network, trace, cfg.training.clients, cfg.orbit.mission_seconds)
    age, counts, summaries = [0] * cfg.training.clients, [0] * cfg.training.clients, []
    steps = accepted = 0
    for round_id in range(cfg.training.rounds):
        start = round_id * cfg.network.round_seconds
        engine, tasks, summary = plan_outage_round(engine, cfg.training.clients, start,
            min(start + cfg.network.round_seconds, cfg.orbit.mission_seconds), 6820712,
            cfg.training.device_step_seconds, cfg.training.local_steps, cfg.network.policy, age, round_id)
        chosen = set(summary["accepted_clients"])
        age = [0 if c in chosen else age[c] + 1 for c in range(cfg.training.clients)]
        counts = [counts[c] + (c in chosen) for c in range(cfg.training.clients)]
        steps += summary["scheduled_steps"]
        accepted += summary["accepted_steps"]
        summaries.append(summary)
    checked = audit(engine, cfg.orbit.mission_seconds)
    failed_bytes = sum(a["wire_bytes"] for a in engine.attempts if not a["success"])
    active_compute = sum(e["end"] - e["start"] for e in engine.compute_events)
    active_radio = sum(a["finish"] - a["start"] for a in engine.attempts)
    return {"performed_steps": steps, "accepted_steps": accepted, "accepted_updates": sum(counts),
            "accepted_per_client": counts, "jain_fairness": sum(counts)**2 / (len(counts) * sum(c*c for c in counts)) if any(counts) else 0,
            "wire_bytes": engine.wire_bytes, "failed_wire_bytes": failed_bytes,
            "failed_attempts": checked["failed_attempts"], "rounds": summaries,
            "successful_rounds": sum(r["successful"] for r in summaries),
            "wasted_compute_seconds": sum(r["wasted_compute_seconds"] for r in summaries),
            "active_compute_seconds": active_compute, "active_radio_seconds": active_radio,
            "energy_sensitivity_wh": {"low_assumed_power": (active_compute * 10 + active_radio * 5) / 3600,
                                      "high_assumed_power": (active_compute * 50 + active_radio * 20) / 3600},
            "pending_queue_upper_bound_per_direction_bytes": cfg.training.clients * 6820712,
            "audit": checked, "mission_clock": engine.mission_clock,
            "decisions": engine.decisions, "observations": engine.observations,
            "attempts": engine.attempts, "compute_events": engine.compute_events,
            "meaning": "CPU scheduling execution only; completed steps are planned for later actual training, not GPU-executed quality."}


def base_config():
    values = load("configs/orbital_sft/five_gpu_one_epoch.yaml").model_dump()
    values["training"].update(clients=5, workers=5, rounds=6, local_steps=4, accumulation=4,
        full_shard_pass=False, epoch_number=1, parent_run=None, evaluate_test=False,
        validation_examples=32, run_seed=42, compatibility_run="results/orbital/compatibility-outages-v2")
    values["orbit"]["mission_seconds"] = 86400
    values["orbit"]["stations"][1] = {"name": "hypothetical-west-ground", "latitude": 34.0, "longitude": -118.0, "elevation_m": 100}
    values["network"].update(policy="fixed", rate_bps=1000000, round_seconds=14400, min_participants=2)
    return values


def geometry(index):
    frozen_source = source_identity()
    altitude, sites = list(itertools.product((450, 550, 700), (1, 2)))[index]
    values = base_config()
    values["orbit"]["altitude_km"] = altitude
    values["orbit"]["stations"] = values["orbit"]["stations"][:sites]
    if len(values["orbit"]["stations"]) != sites:
        raise ValueError("Two documented station profiles required")
    cfg = Settings.model_validate(values)
    output = OUTPUT / "cpu" / f"geometry-{index}"
    output.mkdir(parents=True, exist_ok=False)
    contacts_path = generate(cfg)
    data = read(contacts_path)
    atomic_json(output / "contacts.json", data)
    atomic_json(output / "manifest.json", {"config": cfg.model_dump(), "source_sha256": frozen_source,
                                          "mode": "CPU orbital outage feasibility"})
    entries = []
    for case_index, (rate, step, profile, seed) in enumerate(itertools.product((256000, 1000000), (30, 60), ("none", "low", "high"), (101, 102, 103))):
        trace = make_trace(profile, seed, [s.name for s in cfg.orbit.stations], 5, 86400)
        trace_path = output / f"trace-{profile}-{seed}.json"
        atomic_json(trace_path, trace)
        case = f"g{index}-c{case_index:03d}"
        for policy in POLICIES:
            trial = cfg.model_copy(deep=True)
            trial.network.rate_bps, trial.network.policy = rate, policy
            trial.training.device_step_seconds = step
            trial.training.device_profile = f"UNCALIBRATED sensitivity assumption: {step} simulated seconds per optimizer step"
            result = simulate(trial, data["contacts"], trace)
            details = output / f"{case}-{policy}.json"
            atomic_json(details, result)
            entries.append({"case": case, "geometry": index, "altitude_km": altitude, "station_count": sites,
                "rate_bps": rate, "step_seconds": step, "profile": profile, "fault_seed": seed, "policy": policy,
                "trace": str(trace_path.relative_to(ROOT)), "trace_sha256": digest(trace_path),
                "details": str(details.relative_to(ROOT)),
                **{k: v for k, v in result.items() if k not in {"rounds", "decisions", "observations", "attempts", "compute_events", "meaning"}}})
    if source_identity() != frozen_source:
        raise ValueError("Source changed during CPU sweep")
    atomic_json(output / "results.json", {"geometry": index, "rows": entries, "source_sha256": frozen_source})
    print(f"Geometry {index}: completed and audited {len(entries)} CPU trajectories")


def summarize():
    values = [read(OUTPUT / "cpu" / f"geometry-{i}" / "results.json") for i in range(6)]
    if any(v["source_sha256"] != source_identity() for v in values):
        raise ValueError("CPU runs did not use the current common source")
    rows = [r for v in values for r in v["rows"]]
    if len(rows) != 648:
        raise ValueError("Incomplete CPU outage matrix")
    no_fault = {(r["geometry"], r["rate_bps"], r["step_seconds"]): r for r in rows if r["policy"] == "fixed" and r["profile"] == "none" and r["fault_seed"] == 101}
    candidates = []
    for row in rows:
        if row["policy"] != "fixed" or row["profile"] == "none":
            continue
        control = no_fault[(row["geometry"], row["rate_bps"], row["step_seconds"])]
        if control["accepted_steps"]:
            ratio = row["accepted_steps"] / control["accepted_steps"]
            if 0.2 <= ratio <= 0.8 and row["accepted_steps"] >= 8 and row["failed_attempts"] > 0:
                candidates.append((abs(ratio - 0.5), row["case"], row, control, ratio))
    if not candidates:
        raise ValueError("No scenario satisfies the frozen baseline-pressure rule; do not launch a substituted GPU case")
    _, _, selected, control, ratio = min(candidates, key=lambda v: v[:2])
    geom = read(OUTPUT / "cpu" / f"geometry-{selected['geometry']}" / "manifest.json")["config"]
    geom["network"]["rate_bps"] = selected["rate_bps"]
    geom["training"]["device_step_seconds"] = selected["step_seconds"]
    geom["training"]["device_profile"] = f"UNCALIBRATED sensitivity assumption: {selected['step_seconds']} simulated seconds per optimizer step"
    config_dir = project_path("configs/orbital_sft/outages_v1")
    config_dir.mkdir(exist_ok=True)
    runs = []
    for index, policy in enumerate(("fixed", *POLICIES)):
        label = "no_outage_control" if index == 0 else policy
        trial = json.loads(json.dumps(geom))
        trial["network"]["policy"] = policy
        config_file = config_dir / f"{label}.yaml"
        atomic_json(config_file, trial)
        expected = control if index == 0 else next(r for r in rows if r["case"] == selected["case"] and r["policy"] == policy)
        runs.append({"index": index, "label": label, "policy": policy, "run_id": f"outages-v1-{label}",
                     "config": str(config_file.relative_to(ROOT)), "trace": expected["trace"],
                     "expected_cpu": expected, "config_hash": Settings.model_validate(trial).fingerprint()})
    pilot = load("configs/orbital_sft/compatibility.yaml").model_dump()
    pilot["training"]["compatibility_run"] = "results/orbital/compatibility-outages-v2"
    atomic_json(config_dir / "pilot.yaml", pilot)
    empirical = []
    for result_file in project_path("results/orbital/mission-repeat-v1-s42-connected").glob("round-*/client-*/result.json"):
        entry = read(result_file)
        if entry.get("steps", 0) and "seconds" in entry:
            empirical.append(entry["seconds"] / entry["steps"])
    summary = {"status": "passed", "rows": rows, "selected": selected, "selected_fixed_retention_ratio": ratio,
        "selection_rule": "Only fixed/no-outage control outcomes: accepted-work ratio in [0.2,0.8], >=8 accepted steps and >=1 failed attempt; closest to 0.5, then case ID. Candidate policy outcomes and model quality are not selection inputs.",
        "gpu_runs": runs, "source_sha256": source_identity(), "training_seed": 42,
        "calibration": {"earth_a100_seconds_per_optimizer_step_samples": len(empirical),
                        "earth_a100_median_seconds_per_step": statistics.median(empirical) if empirical else None,
                        "note": "Earth A100 measurements are a reference, not onboard calibration. 30/60 s step profiles and 256 kbps/1 Mbps radio rates are assumptions."},
        "energy_scope": "Posthoc client-active energy sensitivity: compute 10/50 W, radio 5/20 W. No energy-aware admission, calibrated spacecraft power or battery/thermal feasibility claim.",
        "limits": "CPU-only planned execution; one selected scenario and one training seed for bounded GPU confirmation. No model-quality inference from CPU results."}
    atomic_json(OUTPUT / "cpu_summary.json", summary)
    atomic_json(project_path("configs/orbital_sft/outages_selection.json"), {k: summary[k] for k in ("selected", "selected_fixed_retention_ratio", "selection_rule", "gpu_runs", "source_sha256")})
    print(json.dumps({"trajectories": len(rows), "selected_case": selected["case"], "fixed_retention": ratio,
                      "gpu_conditions": [r["label"] for r in runs]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", type=int, choices=range(6))
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    elif args.geometry is not None:
        geometry(args.geometry)
    else:
        parser.error("Supply --geometry or --summarize")


if __name__ == "__main__":
    main()
